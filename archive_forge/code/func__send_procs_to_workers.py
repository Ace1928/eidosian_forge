import os
import multiprocessing as mp
from multiprocessing import Pool, cpu_count, pool
from traceback import format_exception
import sys
from logging import INFO
import gc
from copy import deepcopy
import numpy as np
from ... import logging
from ...utils.profiler import get_system_total_memory_gb
from ..engine import MapNode
from .base import DistributedPluginBase
def _send_procs_to_workers(self, updatehash=False, graph=None):
    """
        Sends jobs to workers when system resources are available.
        """
    jobids = np.flatnonzero(~self.proc_done & (self.depidx.sum(axis=0) == 0).__array__())
    free_memory_gb, free_processors = self._check_resources(self.pending_tasks)
    stats = (len(self.pending_tasks), len(jobids), free_memory_gb, self.memory_gb, free_processors, self.processors)
    if self._stats != stats:
        tasks_list_msg = ''
        if logger.level <= INFO:
            running_tasks = ['  * %s' % self.procs[jobid].fullname for _, jobid in self.pending_tasks]
            if running_tasks:
                tasks_list_msg = '\nCurrently running:\n'
                tasks_list_msg += '\n'.join(running_tasks)
                tasks_list_msg = indent(tasks_list_msg, ' ' * 21)
        logger.info('[LegacyMultiProc] Running %d tasks, and %d jobs ready. Free memory (GB): %0.2f/%0.2f, Free processors: %d/%d.%s', len(self.pending_tasks), len(jobids), free_memory_gb, self.memory_gb, free_processors, self.processors, tasks_list_msg)
        self._stats = stats
    if free_memory_gb < 0.01 or free_processors == 0:
        logger.debug('No resources available')
        return
    if len(jobids) + len(self.pending_tasks) == 0:
        logger.debug('No tasks are being run, and no jobs can be submitted to the queue. Potential deadlock')
        return
    jobids = self._sort_jobs(jobids, scheduler=self.plugin_args.get('scheduler'))
    gc.collect()
    for jobid in jobids:
        if isinstance(self.procs[jobid], MapNode):
            try:
                num_subnodes = self.procs[jobid].num_subnodes()
            except Exception:
                traceback = format_exception(*sys.exc_info())
                self._clean_queue(jobid, graph, result={'result': None, 'traceback': traceback})
                self.proc_pending[jobid] = False
                continue
            if num_subnodes > 1:
                submit = self._submit_mapnode(jobid)
                if not submit:
                    continue
        next_job_gb = min(self.procs[jobid].mem_gb, self.memory_gb)
        next_job_th = min(self.procs[jobid].n_procs, self.processors)
        if next_job_th > free_processors or next_job_gb > free_memory_gb:
            logger.debug('Cannot allocate job %d (%0.2fGB, %d threads).', jobid, next_job_gb, next_job_th)
            continue
        free_memory_gb -= next_job_gb
        free_processors -= next_job_th
        logger.debug('Allocating %s ID=%d (%0.2fGB, %d threads). Free: %0.2fGB, %d threads.', self.procs[jobid].fullname, jobid, next_job_gb, next_job_th, free_memory_gb, free_processors)
        self.proc_done[jobid] = True
        self.proc_pending[jobid] = True
        if self._local_hash_check(jobid, graph):
            continue
        if updatehash or self.procs[jobid].run_without_submitting:
            logger.debug('Running node %s on master thread', self.procs[jobid])
            try:
                self.procs[jobid].run(updatehash=updatehash)
            except Exception:
                traceback = format_exception(*sys.exc_info())
                self._clean_queue(jobid, graph, result={'result': None, 'traceback': traceback})
            self._task_finished_cb(jobid)
            self._remove_node_dirs()
            free_memory_gb += next_job_gb
            free_processors += next_job_th
            self._stats = None
            gc.collect()
            continue
        if self._status_callback:
            self._status_callback(self.procs[jobid], 'start')
        tid = self._submit_job(deepcopy(self.procs[jobid]), updatehash=updatehash)
        if tid is None:
            self.proc_done[jobid] = False
            self.proc_pending[jobid] = False
        else:
            self.pending_tasks.insert(0, (tid, jobid))
        self._stats = None