import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def _save_search(self, row, columns, constraints, name, desc, sharing):
    self._intf._get_entry_point()
    name = name.replace(' ', '_')
    if sharing == 'private':
        users = [self._intf._user]
    elif sharing == 'public':
        users = []
    elif isinstance(sharing, list):
        users = sharing
    else:
        raise NotSupportedError('Share mode %s not valid' % sharing)
    self._intf._exec('%s/search/saved/%s?inbody=true' % (self._intf._entry, name), method='PUT', body=build_search_document(row, columns, constraints, name, desc.replace('%', '%%'), users))