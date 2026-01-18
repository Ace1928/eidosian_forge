from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
def __to_str_dictionary(self, value, indent):
    strResults = ''
    try:
        for key, val in value.items():
            if isinstance(val, collections.Mapping):
                if len(val.values()) > 0:
                    strResults += '%s%s:\n' % (indent, key)
                    strResults += self.__to_str_dictionary(val, '  ' + indent)
                else:
                    strResults += '%s%s: %s\n' % (indent, key, 'None')
            elif type(val) is list and isinstance(val[0], collections.Mapping):
                for i in range(0, len(val)):
                    strResults += '%s%s: [%d of %d]\n' % (indent, key, i + 1, len(val))
                    strResults += self.__to_str_dictionary(val[i], '  ' + indent)
            else:
                strResults += '%s%s: %s\n' % (indent, key, str(val))
    except Exception as e:
        strResults += '\n[Error] ' + str(e)
    return strResults