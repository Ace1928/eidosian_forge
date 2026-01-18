import re
from _csv import Error, __version__, writer, reader, register_dialect, \
from _csv import Dialect as _Dialect
from io import StringIO
def _dict_to_list(self, rowdict):
    if self.extrasaction == 'raise':
        wrong_fields = rowdict.keys() - self.fieldnames
        if wrong_fields:
            raise ValueError('dict contains fields not in fieldnames: ' + ', '.join([repr(x) for x in wrong_fields]))
    return (rowdict.get(key, self.restval) for key in self.fieldnames)