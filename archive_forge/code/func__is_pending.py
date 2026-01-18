import os
import stat
from time import sleep
import subprocess
import simplejson as json
from ... import logging
from ...interfaces.base import CommandLine
from .base import SGELikeBatchManagerBase, logger
def _is_pending(self, taskid):
    proc = subprocess.Popen(['oarstat', '-J', '-s', '-j', taskid], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    o, e = proc.communicate()
    parsed_result = json.loads(o)[taskid].lower()
    is_pending = 'error' not in parsed_result and 'terminated' not in parsed_result
    return is_pending