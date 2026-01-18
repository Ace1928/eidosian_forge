import os
import pwd
import re
import subprocess
import time
import xml.dom.minidom
import random
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
class QJobInfo(object):
    """Information about a single job created by OGE/SGE or similar
    Each job is responsible for knowing it's own refresh state
    :author Hans J. Johnson
    """

    def __init__(self, job_num, job_queue_state, job_time, job_queue_name, job_slots, qsub_command_line):
        self._job_num = int(job_num)
        self._job_queue_state = str(job_queue_state)
        self._job_time = job_time
        self._job_info_creation_time = time.time()
        self._job_queue_name = job_queue_name
        self._job_slots = int(job_slots)
        self._qsub_command_line = qsub_command_line

    def __repr__(self):
        return '{:<8d}{:12}{:<3d}{:20}{:8}{}'.format(self._job_num, self._job_queue_state, self._job_slots, time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(self._job_time)), self._job_queue_name, self._qsub_command_line)

    def is_initializing(self):
        return self._job_queue_state == 'initializing'

    def is_zombie(self):
        return self._job_queue_state == 'zombie' or self._job_queue_state == 'finished'

    def is_running(self):
        return self._job_queue_state == 'running'

    def is_pending(self):
        return self._job_queue_state == 'pending'

    def is_job_state_pending(self):
        """Return True, unless job is in the "zombie" status"""
        time_diff = time.time() - self._job_info_creation_time
        if self.is_zombie():
            sge_debug_print("DONE! QJobInfo.IsPending found in 'zombie' list, returning False so claiming done!\n{0}".format(self))
            is_pending_status = False
        elif self.is_initializing() and time_diff > 600:
            sge_debug_print("FAILURE! QJobInfo.IsPending found long running at {1} seconds'initializing' returning False for to break loop!\n{0}".format(self, time_diff))
            is_pending_status = True
        else:
            is_pending_status = True
        return is_pending_status

    def update_info(self, job_queue_state, job_time, job_queue_name, job_slots):
        self._job_queue_state = job_queue_state
        self._job_time = job_time
        self._job_queue_name = job_queue_name
        self._job_slots = int(job_slots)

    def set_state(self, new_state):
        self._job_queue_state = new_state