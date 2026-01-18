import re
import sys
def ConvertToMSBuild(self, value):
    index = int(value)
    if index < 0 or index >= len(self._label_list):
        raise ValueError('index value (%d) not in expected range [0, %d)' % (index, len(self._label_list)))
    label = self._label_list[index]
    if label is None:
        raise ValueError('converted value for %s not specified.' % value)
    return label