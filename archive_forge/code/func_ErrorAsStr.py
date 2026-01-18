from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pipes
from fire import inspectutils
def ErrorAsStr(self):
    return ' '.join((str(arg) for arg in self._error.args))