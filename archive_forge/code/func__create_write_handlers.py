import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def _create_write_handlers(self, hub, pack=pack, dumps=_pickle.dumps, protocol=HIGHEST_PROTOCOL):
    """Create handlers used to write data to child processes."""
    fileno_to_inq = self._fileno_to_inq
    fileno_to_synq = self._fileno_to_synq
    outbound = self.outbound_buffer
    pop_message = outbound.popleft
    put_message = outbound.append
    all_inqueues = self._all_inqueues
    active_writes = self._active_writes
    active_writers = self._active_writers
    busy_workers = self._busy_workers
    diff = all_inqueues.difference
    add_writer = hub.add_writer
    hub_add, hub_remove = (hub.add, hub.remove)
    mark_write_fd_as_active = active_writes.add
    mark_write_gen_as_active = active_writers.add
    mark_worker_as_busy = busy_workers.add
    write_generator_done = active_writers.discard
    get_job = self._cache.__getitem__
    write_stats = self.write_stats
    is_fair_strategy = self.sched_strategy == SCHED_STRATEGY_FAIR
    revoked_tasks = worker_state.revoked
    getpid = os.getpid
    precalc = {ACK: self._create_payload(ACK, (0,)), NACK: self._create_payload(NACK, (0,))}

    def _put_back(job, _time=time.time):
        if job._terminated is not None or job.correlation_id in revoked_tasks:
            if not job._accepted:
                job._ack(None, _time(), getpid(), None)
            job._set_terminated(job._terminated)
        elif job not in outbound:
            outbound.appendleft(job)
    self._put_back = _put_back

    def on_poll_start():
        inactive = diff(active_writes)
        if is_fair_strategy:
            add_cond = outbound and len(busy_workers) < len(all_inqueues)
        else:
            add_cond = outbound
        if add_cond:
            iterate_file_descriptors_safely(inactive, all_inqueues, hub_add, None, WRITE | ERR, consolidate=True)
        else:
            iterate_file_descriptors_safely(inactive, all_inqueues, hub_remove)
    self.on_poll_start = on_poll_start

    def on_inqueue_close(fd, proc):
        busy_workers.discard(fd)
        try:
            if fileno_to_inq[fd] is proc:
                fileno_to_inq.pop(fd, None)
                active_writes.discard(fd)
                all_inqueues.discard(fd)
        except KeyError:
            pass
    self.on_inqueue_close = on_inqueue_close
    self.hub_remove = hub_remove

    def schedule_writes(ready_fds, total_write_count=None):
        if not total_write_count:
            total_write_count = [0]
        num_ready = len(ready_fds)
        for _ in range(num_ready):
            ready_fd = ready_fds[total_write_count[0] % num_ready]
            total_write_count[0] += 1
            if ready_fd in active_writes:
                continue
            if is_fair_strategy and ready_fd in busy_workers:
                continue
            if ready_fd not in all_inqueues:
                hub_remove(ready_fd)
                continue
            try:
                job = pop_message()
            except IndexError:
                for inqfd in diff(active_writes):
                    hub_remove(inqfd)
                break
            else:
                if not job._accepted:
                    try:
                        proc = job._scheduled_for = fileno_to_inq[ready_fd]
                    except KeyError:
                        put_message(job)
                        continue
                    cor = _write_job(proc, ready_fd, job)
                    job._writer = ref(cor)
                    mark_write_gen_as_active(cor)
                    mark_write_fd_as_active(ready_fd)
                    mark_worker_as_busy(ready_fd)
                    try:
                        next(cor)
                    except StopIteration:
                        pass
                    except OSError as exc:
                        if exc.errno != errno.EBADF:
                            raise
                    else:
                        add_writer(ready_fd, cor)
    hub.consolidate_callback = schedule_writes

    def send_job(tup):
        body = dumps(tup, protocol=protocol)
        body_size = len(body)
        header = pack('>I', body_size)
        job = get_job(tup[1][0])
        job._payload = (memoryview(header), memoryview(body), body_size)
        put_message(job)
    self._quick_put = send_job

    def on_not_recovering(proc, fd, job, exc):
        logger.exception('Process inqueue damaged: %r %r: %r', proc, proc.exitcode, exc)
        if proc._is_alive():
            proc.terminate()
        hub.remove(fd)
        self._put_back(job)

    def _write_job(proc, fd, job):
        header, body, body_size = job._payload
        errors = 0
        try:
            job._write_to = proc
            send = proc.send_job_offset
            Hw = Bw = 0
            while Hw < 4:
                try:
                    Hw += send(header, Hw)
                except Exception as exc:
                    if getattr(exc, 'errno', None) not in UNAVAIL:
                        raise
                    errors += 1
                    if errors > 100:
                        on_not_recovering(proc, fd, job, exc)
                        raise StopIteration()
                    yield
                else:
                    errors = 0
            while Bw < body_size:
                try:
                    Bw += send(body, Bw)
                except Exception as exc:
                    if getattr(exc, 'errno', None) not in UNAVAIL:
                        raise
                    errors += 1
                    if errors > 100:
                        on_not_recovering(proc, fd, job, exc)
                        raise StopIteration()
                    yield
                else:
                    errors = 0
        finally:
            hub_remove(fd)
            write_stats[proc.index] += 1
            active_writes.discard(fd)
            write_generator_done(job._writer())

    def send_ack(response, pid, job, fd):
        msg = Ack(job, fd, precalc[response])
        callback = promise(write_generator_done)
        cor = _write_ack(fd, msg, callback=callback)
        mark_write_gen_as_active(cor)
        mark_write_fd_as_active(fd)
        callback.args = (cor,)
        add_writer(fd, cor)
    self.send_ack = send_ack

    def _write_ack(fd, ack, callback=None):
        header, body, body_size = ack[2]
        try:
            try:
                proc = fileno_to_synq[fd]
            except KeyError:
                raise StopIteration()
            send = proc.send_syn_offset
            Hw = Bw = 0
            while Hw < 4:
                try:
                    Hw += send(header, Hw)
                except Exception as exc:
                    if getattr(exc, 'errno', None) not in UNAVAIL:
                        raise
                    yield
            while Bw < body_size:
                try:
                    Bw += send(body, Bw)
                except Exception as exc:
                    if getattr(exc, 'errno', None) not in UNAVAIL:
                        raise
                    yield
        finally:
            if callback:
                callback()
            active_writes.discard(fd)