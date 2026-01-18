import _thread
import codecs
import operator
import os
import pickle
import sys
import threading
from typing import Dict, TextIO
from _lsprof import Profiler, profiler_entry
from . import errors
def _subentry(self, lineno, subentry):
    out_file = self.out_file
    code = subentry.code
    totaltime = int(subentry.totaltime * 1000)
    if isinstance(code, str):
        out_file.write('cfi=~\n')
        out_file.write('cfn={}\n'.format(label(code, True)))
        out_file.write('calls=%d 0\n' % (subentry.callcount,))
    else:
        out_file.write('cfi={}\n'.format(code.co_filename))
        out_file.write('cfn={}\n'.format(label(code, True)))
        out_file.write('calls=%d %d\n' % (subentry.callcount, code.co_firstlineno))
    out_file.write('%d %d\n' % (lineno, totaltime))