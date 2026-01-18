from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
def __to_str(self, results):
    strResults = ''
    indent = '  '
    for key, val in results.items():
        if type(val) is list:
            for i in range(0, len(val)):
                strResults += '%s%s: [%d of %d]\n' % (indent, key, i + 1, len(val))
                strResults += self.__to_str_dictionary(val[i], '  ' + indent)
        else:
            strResults += '%s%s: %s\n' % (indent, key, str(val))
    return strResults