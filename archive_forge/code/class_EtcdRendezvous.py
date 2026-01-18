import json
import logging
import sys
import threading
import time
from typing import Optional
import etcd  # type: ignore[import]
from torch.distributed.elastic.rendezvous import (
from .utils import parse_rendezvous_endpoint
from .etcd_store import EtcdStore, cas_delay
class EtcdRendezvous:
    """A rendezvous implementation that uses `etcd <https://etcd.io/>`__ as the backend store."""

    def __init__(self, client, prefix, run_id, num_min_workers, num_max_workers, timeout, last_call_timeout):
        self.client = client
        log.info('Etcd machines: %s', self.client.machines)
        self._prefix = prefix
        self._run_id = run_id
        self._num_min_workers = num_min_workers
        self._num_max_workers = num_max_workers
        self._timeout = timeout
        self._last_call_timeout = last_call_timeout
        self._lease_run_id_stop = None
        self._lease_this_rank_stop = None
        if not self._prefix.endswith('/'):
            self._prefix += '/'
        if self._prefix != '/':
            self.create_path_if_not_exists(self._prefix)
        self.create_path_if_not_exists(self.get_path(''), ttl=CONST_RUNID_SUBROOT_TTL)
        self._lease_run_id_stop = self.setup_lease_renewal(self.get_path(''), ttl=CONST_RUNID_SUBROOT_TTL)
        self.create_path_if_not_exists(self.get_path('/rdzv'))
        try:
            self.client.write(key=self.get_path('/rdzv/version_counter'), value='0', prevExist=False)
        except etcd.EtcdAlreadyExist:
            pass

    def __del__(self):
        if self._lease_run_id_stop is not None:
            self._lease_run_id_stop.set()
        if self._lease_this_rank_stop is not None:
            self._lease_this_rank_stop.set()

    def rendezvous_barrier(self):
        """
        Main entry point for next rendezvous.

        This method is blocking until rendezvous succeeds or a timeout occurs.

        Returns:
             ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousTimeoutError - timeout waiting for rendezvous
            RendezvousClosedError - rendezvous is or was closed while waiting
            RendezvousError - other persistent errors that
             render the rendezvous non-retryable
        """
        self._rendezvous_deadline = time.time() + self._timeout
        while True:
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError()
            log.info('Attempting to join next rendezvous')
            try:
                if self._lease_this_rank_stop is not None:
                    self._lease_this_rank_stop.set()
                return self.init_phase()
            except EtcdRendezvousRetryImmediately:
                pass
            except EtcdRendezvousRetryableFailure:
                time.sleep(1)
            except RendezvousTimeoutError:
                log.info('Rendezvous timeout occurred in EtcdRendezvousHandler')
                raise
            except RendezvousClosedError:
                log.info('Rendezvous for run_id=%s was observed to be closed', self._run_id)
                raise
            except RendezvousError:
                raise
            except Exception as e:
                log.info('Rendezvous attempt failed, will retry. Reason: %s', e)
                time.sleep(1)

    def init_phase(self):
        """
        Initially, the rendezvous state is expected to be one of:

        1. empty (non-existent) - in this case we try to create a new one.
        2. joinable - we try to join it.
        3. final - we announce ourselves as waiting, and go into monitoring mode

        Any other state is considered transitional, and will be retried after
        a short delay.

        Returns:
            ``(rdzv_version, rank, world_size)``

        Raises:
            RendezvousClosedError - current rendezvous was/is closed
            EtcdRendezvousRetryableFailure - observed some intermediate
             state, which is best handled by retrying later
        """
        try:
            active_version = self.try_create_rendezvous()
            state = json.loads(active_version.value)
            log.info('New rendezvous state created: %s', state)
        except etcd.EtcdAlreadyExist:
            active_version, state = self.get_rdzv_state()
            log.info('Observed existing rendezvous state: %s', state)
        if state['status'] == 'closed':
            raise RendezvousClosedError()
        if state['status'] == 'joinable':
            return self.join_phase(state['version'])
        if state['status'] == 'final':
            self.handle_existing_rendezvous(state['version'])
            raise EtcdRendezvousRetryImmediately()
        self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
        raise EtcdRendezvousRetryableFailure()

    def join_phase(self, expected_version):
        """
        We observed a rendezvous state in 'joinable' state, and attempt to join this
        particular version, and then wait for all other peers to join.
        """
        active_version, this_rank = self.join_rendezvous(expected_version)
        state = json.loads(active_version.value)
        log.info('Joined rendezvous version %s as rank %s. Full state: %s', state['version'], this_rank, state)
        if this_rank == self._num_min_workers - 1 and state['status'] == 'joinable':
            log.info('Rank %s is responsible for join last call.', this_rank)
            last_call_deadline = time.time() + self._last_call_timeout
            self.handle_join_last_call(expected_version, last_call_deadline)
            log.info('Rank %s finished join last call.', this_rank)
        log.info('Waiting for remaining peers.')
        active_version = self.wait_for_peers(expected_version)
        state = json.loads(active_version.value)
        assert state['version'] == expected_version, 'Logic error: failed to observe version mismatch'
        return self.confirm_phase(expected_version, this_rank)

    def confirm_phase(self, expected_version, this_rank):
        """
        Once the rendezvous state transitions from 'joinable' to 'frozen',
        we have every participant confirm their membership and setup per-member
        keep-alive TTL keys, and then wait for all other participants to confirm,
        which would then successfully conclude this rendezvous.
        """
        log.info('All peers arrived. Confirming membership.')
        self.confirm_membership(expected_version, this_rank)
        log.info('Waiting for confirmations from all peers.')
        active_version = self.wait_for_final(expected_version)
        state = json.loads(active_version.value)
        log.info('Rendezvous version %s is complete. Final state: %s', state['version'], state)
        return (state['version'], this_rank, len(state['participants']))

    def handle_existing_rendezvous(self, expected_version):
        """
        Handle the case when there's an existing (state 'final) rendezvous already
        in place, and we have to announce ourselves waiting, and wait until
        the next rendezvous opportunity.
        """
        active_state = self.announce_self_waiting(expected_version)
        log.info('Added self to waiting list. Rendezvous full state: %s', active_state.value)
        self.wait_for_rendezvous_to_free(expected_version)
        log.info('Previously existing rendezvous state changed. Will re-try joining.')

    def try_create_rendezvous(self):
        """
        Create new rendezvous state or raise an exception that indicates an unexpected state (e.g. already exists).

        Raises:
             RendezvousError - on unexpected state
        """
        active_version = self.client.write(key=self.get_path('/rdzv/active_version'), value=json.dumps({'status': 'setup'}), prevExist=False, ttl=CONST_ETCD_SETUP_TTL)
        try:
            version_counter = self.client.get(self.get_path('/rdzv/version_counter'))
            version_counter.value = str(int(version_counter.value) + 1)
            self.client.update(version_counter)
        except (etcd.EtcdKeyNotFound, etcd.EtcdCompareFailed) as e:
            raise RendezvousError('Unexpected state of EtcdRendezvousHandler, worker needs to die.') from e
        self.client.write(key=self.get_path(f'/rdzv/v_{version_counter.value}'), value=None, dir=True, prevExist=False)
        return self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps({'status': 'joinable', 'version': version_counter.value, 'participants': []}), prev_value=active_version.value)

    def join_rendezvous(self, expected_version):
        """Helper method for the join phase."""
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()
            if state['status'] != 'joinable':
                raise EtcdRendezvousRetryableFailure('Rendezvous state became non-joinable before we could join. Must join next one.')
            if state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
            assert len(state['participants']) < self._num_max_workers, 'Logic error: joinable rendezvous should always have space left'
            this_rank = len(state['participants'])
            state['participants'].append(this_rank)
            set_ttl: Optional[int] = None
            if len(state['participants']) == self._num_max_workers:
                state['status'] = 'frozen'
                state['keep_alives'] = []
                set_ttl = CONST_ETCD_FROZEN_TTL
            elif len(state['participants']) >= self._num_min_workers:
                set_ttl = CONST_ETCD_JOINABLE_EPHEMERAL_TTL
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=set_ttl)
                return (active_version, this_rank)
            except etcd.EtcdCompareFailed:
                log.info('Join rendezvous CAS unsuccessful, retrying')

    def wait_for_peers(self, expected_version):
        """Helper method for the join phase."""
        active_version, state = self.get_rdzv_state()
        while True:
            if state['status'] == 'frozen' and state['version'] == expected_version:
                return active_version
            elif state['status'] == 'joinable' and state['version'] == expected_version:
                active_version, state = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
            else:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')

    def confirm_membership(self, expected_version, this_rank):
        """Helper method for the confirm phase."""
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()
            if state['status'] != 'frozen':
                raise EtcdRendezvousRetryImmediately('Rendezvous no longer frozen, before we confirmed. Must join next one')
            if state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately('Rendezvous version changed. Must try join the new one.')
            this_lease_key = self.get_path(f'/rdzv/v_{expected_version}/rank_{this_rank}')
            self.client.set(this_lease_key, value=None, ttl=CONST_WORKER_KEEPALIVE_TTL)
            state['keep_alives'].append(this_lease_key)
            if len(state['keep_alives']) == len(state['participants']):
                state['status'] = 'final'
                state['num_workers_waiting'] = 0
                finalize = True
            else:
                finalize = False
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=None if finalize else CONST_ETCD_FROZEN_TTL)
                self._lease_this_rank_stop = self.setup_lease_renewal(this_lease_key, ttl=CONST_WORKER_KEEPALIVE_TTL)
                return active_version
            except etcd.EtcdCompareFailed:
                log.info('Confirm membership CAS unsuccessful, retrying')

    def wait_for_final(self, expected_version):
        """Helper method for the confirm phase."""
        active_version, state = self.get_rdzv_state()
        while True:
            if state['status'] == 'final' and state['version'] == expected_version:
                return active_version
            elif state['status'] == 'frozen' and state['version'] == expected_version:
                active_version, state = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1)
            else:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')

    def announce_self_waiting(self, expected_version):
        """
        Announce this worker is waiting (via num_workers_waiting counter) to join next
        rendezvous, but only if state and version match.
        """
        while True:
            cas_delay()
            active_version, state = self.get_rdzv_state()
            if state['status'] != 'final' or state['version'] != expected_version:
                raise EtcdRendezvousRetryImmediately()
            state['num_workers_waiting'] += 1
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value)
                return active_version
            except etcd.EtcdCompareFailed:
                log.info('Announce self as waiting CAS unsuccessful, retrying')

    def wait_for_rendezvous_to_free(self, expected_version):
        """
        When there's an existing valid rendezvous in state 'final', we have to wait until the next opportunity to join.

        Such opportunity may come from:

        1. rendezvous state changed by someone else, in which case we unblock and retry.
        2. rendezvous becomes invalid because at least one member failed to renew their
           leased keep_alive node. We detect this, and destroy the rendezvous.
        """
        active_version, state = self.get_rdzv_state()
        while True:
            if state['status'] != 'final' or state['version'] != expected_version:
                return
            alive_members = self.client.get(self.get_path(f'/rdzv/v_{expected_version}'))
            keep_alive_keys = [ch.key for ch in alive_members.children]
            for key in state['keep_alives']:
                if key not in keep_alive_keys:
                    log.info('Keep-alive key %s is not renewed.', key)
                    log.info('Rendezvous version %s is incomplete. ', expected_version)
                    log.info('Attempting to destroy it.')
                    self.client.delete(key=self.get_path('/rdzv/active_version'), prevValue=active_version.value)
                    log.info('Destroyed rendezvous version %s successfully.', expected_version)
                    return
            try:
                overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
                self.client.watch(key=self.get_path('/rdzv'), index=active_version.etcd_index + 1, recursive=True, timeout=overall_timeout)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass
            if time.time() > self._rendezvous_deadline:
                raise RendezvousTimeoutError()
            active_version, state = self.get_rdzv_state()

    def handle_join_last_call(self, expected_version, deadline):
        """
        After we reach min number of workers, one particular worker takes on the
        responsibility of waiting an additional timeout before closing the join window.
        If the worker responsible for this fails, the rendezvous will be destroyed due
        to expiring TTL, and the other participants will re-rendezvous.

        Here we expect to see state <joinable, expected_version>
        Exit gracefully if either:

        1. state becomes <frozen, expected_version>
        2. timeout happens (reaching deadline), in which case
           we try the transition to <frozen, expected_version>

        Exit with exception otherwise.
        """
        active_version, state = self.get_rdzv_state()
        while True:
            if state['status'] == 'frozen' and state['version'] == expected_version:
                return
            if state['status'] != 'joinable' or state['version'] != expected_version:
                raise EtcdRendezvousRetryableFailure('Rendezvous state transition no longer possible. Must re-enter.')
            if time.time() >= deadline:
                state['status'] = 'frozen'
                state['keep_alives'] = []
                try:
                    active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value, ttl=CONST_ETCD_FROZEN_TTL)
                    return
                except etcd.EtcdCompareFailed:
                    log.info('Join last-call transition CAS unsuccessful. Will retry')
                    cas_delay()
                    active_version, state = self.get_rdzv_state()
                    continue
            try:
                active_version = self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=active_version.value, prev_value=active_version.value, ttl=CONST_ETCD_JOINABLE_EPHEMERAL_TTL)
                timeout = min(CONST_ETCD_JOINABLE_EPHEMERAL_TTL / 2, deadline - time.time() + 1.0)
                active_version, state = self.try_wait_for_state_change(etcd_index=active_version.etcd_index + 1, timeout=timeout)
            except etcd.EtcdCompareFailed:
                log.info('Join last-call TTL refresh CAS unsuccessful, will retry')
                cas_delay()
                active_version, state = self.get_rdzv_state()

    def set_closed(self):
        """
        Mark rendezvous 'closed' for current run_id, which is used to signal other
        participants to not attempt to perform (re-)rendezvous. This is useful
        when one of the workers decides the job is complete.
        """
        while True:
            active_version, state = self.get_rdzv_state()
            if state['status'] == 'closed':
                return
            state['status'] = 'closed'
            try:
                self.client.test_and_set(key=self.get_path('/rdzv/active_version'), value=json.dumps(state), prev_value=active_version.value)
                return
            except etcd.EtcdCompareFailed:
                log.info('Set closed CAS unsuccessful, retrying')
                cas_delay()

    def get_rdzv_state(self):
        active_version = self.client.get(key=self.get_path('/rdzv/active_version'))
        return (active_version, json.loads(active_version.value))

    def try_wait_for_state_change(self, etcd_index, timeout=None):
        overall_timeout = max(self._rendezvous_deadline - time.time(), 0.0) + 1.0
        timeout = overall_timeout if timeout is None else min(timeout, overall_timeout)
        try:
            self.client.watch(self.get_path('/rdzv/active_version'), index=etcd_index, timeout=timeout)
        except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
            pass
        if time.time() > self._rendezvous_deadline:
            raise RendezvousTimeoutError()
        return self.get_rdzv_state()

    def get_path(self, path):
        if not path.startswith('/'):
            path = '/' + path
        return f'{self._prefix}run_{self._run_id}{path}'

    def create_path_if_not_exists(self, full_path, ttl=None):
        try:
            self.client.write(key=full_path, value=None, dir=True, prevExist=False, ttl=ttl)
        except etcd.EtcdAlreadyExist:
            pass

    def setup_lease_renewal(self, full_path, ttl):

        def lease_worker(client, path, ttl, stop_event):
            while True:
                try:
                    client.refresh(path, ttl=ttl)
                except etcd.EtcdKeyNotFound:
                    break
                except ConnectionRefusedError:
                    break
                if stop_event.wait(timeout=ttl / 2):
                    break
        lease_stop_event = threading.Event()
        lease_thread = threading.Thread(target=lease_worker, args=(self.client, full_path, ttl, lease_stop_event))
        lease_thread.daemon = True
        lease_thread.start()
        return lease_stop_event

    def store_extra_data(self, rdzv_version, key, value):
        node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
        try:
            extra_data = self.client.write(key=node, value=json.dumps({key: value}), prevExist=False)
            return
        except etcd.EtcdAlreadyExist:
            pass
        while True:
            extra_data = self.client.get(node)
            new_extra_data_value = json.loads(extra_data.value)
            new_extra_data_value[key] = value
            try:
                extra_data = self.client.test_and_set(key=node, value=json.dumps(new_extra_data_value), prev_value=extra_data.value)
                return
            except etcd.EtcdCompareFailed:
                log.info('Store extra_data CAS unsuccessful, retrying')
                time.sleep(0.1)

    def load_extra_data(self, rdzv_version, key, timeout=None):
        node = self.get_path(f'/rdzv/v_{rdzv_version}/extra_data')
        node_dir = self.get_path(f'/rdzv/v_{rdzv_version}')
        while True:
            root = self.client.get(node_dir)
            extra_data = [n for n in root.children if n.key == node]
            assert len(extra_data) <= 1
            if len(extra_data) == 1:
                extra_data_dict = json.loads(extra_data[0].value)
                if key in extra_data_dict:
                    return extra_data_dict[key]
            try:
                self.client.watch(node, index=root.etcd_index + 1)
            except (etcd.EtcdEventIndexCleared, etcd.EtcdWatchTimedOut):
                pass

    def setup_kv_store(self, rdzv_version):
        store_path = self.get_path(f'/rdzv/v_{rdzv_version}/kv')
        self.create_path_if_not_exists(store_path)
        return EtcdStore(etcd_client=self.client, etcd_store_prefix=store_path)