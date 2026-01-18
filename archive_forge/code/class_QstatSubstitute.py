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
class QstatSubstitute(object):
    """A wrapper for Qstat to avoid overloading the
    SGE/OGS server with rapid continuous qstat requests"""

    def __init__(self, qstat_instant_executable='qstat', qstat_cached_executable='qstat'):
        """
        :param qstat_instant_executable:
        :param qstat_cached_executable:
        """
        self._qstat_instant_executable = qstat_instant_executable
        self._qstat_cached_executable = qstat_cached_executable
        self._out_of_scope_jobs = list()
        self._task_dictionary = dict()
        self._remove_old_jobs()

    def _remove_old_jobs(self):
        """This is only called during initialization of the function for the purpose
        of identifying jobs that are not part of this run of nipype.  They
        are jobs that existed prior to starting a new jobs, so they are irrelevant.
        """
        self._run_qstat('QstatInitialization', True)

    def add_startup_job(self, taskid, qsub_command_line):
        """
        :param taskid: The job id
        :param qsub_command_line: When initializing, re-use the job_queue_name
        :return: NONE
        """
        taskid = int(taskid)
        self._task_dictionary[taskid] = QJobInfo(taskid, 'initializing', time.time(), 'noQueue', 1, qsub_command_line)

    @staticmethod
    def _qacct_verified_complete(taskid):
        """request definitive job completion information for the current job
        from the qacct report
        """
        sge_debug_print('WARNING:  CONTACTING qacct for finished jobs, {0}: {1}'.format(time.time(), 'Verifying Completion'))
        this_command = 'qacct'
        qacct_retries = 10
        is_complete = False
        while qacct_retries > 0:
            qacct_retries -= 1
            try:
                proc = subprocess.Popen([this_command, '-o', pwd.getpwuid(os.getuid())[0], '-j', str(taskid)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                qacct_result, _ = proc.communicate()
                if qacct_result.find(str(taskid)):
                    is_complete = True
                sge_debug_print('NOTE: qacct for jobs\n{0}'.format(qacct_result))
                break
            except:
                sge_debug_print('NOTE: qacct call failed')
                time.sleep(5)
                pass
        return is_complete

    def _parse_qstat_job_list(self, xml_job_list):
        current_jobs_parsed = list()
        for current_job_element in xml_job_list:
            try:
                job_queue_name = current_job_element.getElementsByTagName('queue_name')[0].childNodes[0].data
            except:
                job_queue_name = 'unknown'
            try:
                job_slots = int(current_job_element.getElementsByTagName('slots')[0].childNodes[0].data)
            except:
                job_slots = -1
            job_queue_state = current_job_element.getAttribute('state')
            job_num = int(current_job_element.getElementsByTagName('JB_job_number')[0].childNodes[0].data)
            try:
                job_time_text = current_job_element.getElementsByTagName('JAT_start_time')[0].childNodes[0].data
                job_time = float(time.mktime(time.strptime(job_time_text, '%Y-%m-%dT%H:%M:%S')))
            except:
                job_time = float(0.0)
            task_id = int(job_num)
            if task_id in self._task_dictionary:
                self._task_dictionary[task_id].update_info(job_queue_state, job_time, job_queue_name, job_slots)
                sge_debug_print('Updating job:  {0}'.format(self._task_dictionary[task_id]))
                current_jobs_parsed.append(task_id)
            else:
                self._out_of_scope_jobs.append(task_id)
        for dictionary_job in list(self._task_dictionary.keys()):
            if dictionary_job not in current_jobs_parsed:
                is_completed = self._qacct_verified_complete(dictionary_job)
                if is_completed:
                    self._task_dictionary[dictionary_job].set_state('zombie')
                else:
                    sge_debug_print('ERROR:  Job not in current parselist, and not in done list {0}: {1}'.format(dictionary_job, self._task_dictionary[dictionary_job]))
                    pass
            if self._task_dictionary[dictionary_job].is_initializing():
                is_completed = self._qacct_verified_complete(dictionary_job)
                if is_completed:
                    self._task_dictionary[dictionary_job].set_state('zombie')
                else:
                    sge_debug_print('ERROR:  Job not in still in initialization mode, and not in done list {0}: {1}'.format(dictionary_job, self._task_dictionary[dictionary_job]))
                    pass

    def _run_qstat(self, reason_for_qstat, force_instant=True):
        """request all job information for the current user in xmlformat.
        See documentation from java documentation:
        http://arc.liv.ac.uk/SGE/javadocs/jgdi/com/sun/grid/jgdi/monitoring/filter/JobStateFilter.html
        -s r gives running jobs
        -s z gives recently completed jobs (**recently** is very ambiguous)
        -s s suspended jobs
        """
        sge_debug_print('WARNING:  CONTACTING qmaster for jobs, {0}: {1}'.format(time.time(), reason_for_qstat))
        if force_instant:
            this_command = self._qstat_instant_executable
        else:
            this_command = self._qstat_cached_executable
        qstat_retries = 10
        while qstat_retries > 0:
            qstat_retries -= 1
            try:
                proc = subprocess.Popen([this_command, '-u', pwd.getpwuid(os.getuid())[0], '-xml', '-s', 'psrz'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                qstat_xml_result, _ = proc.communicate()
                dom = xml.dom.minidom.parseString(qstat_xml_result)
                jobs = dom.getElementsByTagName('job_info')
                run = jobs[0]
                runjobs = run.getElementsByTagName('job_list')
                self._parse_qstat_job_list(runjobs)
                break
            except Exception as inst:
                exception_message = 'QstatParsingError:\n\t{0}\n\t{1}\n'.format(type(inst), inst)
                sge_debug_print(exception_message)
                time.sleep(5)
                pass

    def print_dictionary(self):
        """For debugging"""
        for vv in list(self._task_dictionary.values()):
            sge_debug_print(str(vv))

    def is_job_pending(self, task_id):
        task_id = int(task_id)
        if task_id in self._task_dictionary:
            job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
            if job_is_pending:
                self._run_qstat('checking job pending status {0}'.format(task_id), False)
                job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
        else:
            self._run_qstat('checking job pending status {0}'.format(task_id), True)
            if task_id in self._task_dictionary:
                job_is_pending = self._task_dictionary[task_id].is_job_state_pending()
            else:
                sge_debug_print('ERROR: Job {0} not in task list, even after forced qstat!'.format(task_id))
                job_is_pending = False
        if not job_is_pending:
            sge_debug_print('DONE! Returning for {0} claiming done!'.format(task_id))
            if task_id in self._task_dictionary:
                sge_debug_print('NOTE: Adding {0} to OutOfScopeJobs list!'.format(task_id))
                self._out_of_scope_jobs.append(int(task_id))
                self._task_dictionary.pop(task_id)
            else:
                sge_debug_print('ERROR: Job {0} not in task list, but attempted to be removed!'.format(task_id))
        return job_is_pending