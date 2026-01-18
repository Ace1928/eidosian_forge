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