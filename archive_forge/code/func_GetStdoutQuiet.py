import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetStdoutQuiet(cmdlist):
    """Returns the content of standard output returned by invoking |cmdlist|.
  Ignores the stderr.
  Raises |GypError| if the command return with a non-zero return code."""
    job = subprocess.Popen(cmdlist, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out = job.communicate()[0].decode('utf-8')
    if job.returncode != 0:
        raise GypError('Error %d running %s' % (job.returncode, cmdlist[0]))
    return out.rstrip('\n')