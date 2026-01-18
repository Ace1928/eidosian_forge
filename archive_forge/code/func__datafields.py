from . import schema
from .jsonutil import get_column
from .search import Search
def _datafields(self, datatype, pattern='*', prepend_type=True):
    self._intf._get_entry_point()
    search_fds = self._get_json('%s/search/elements/%s?format=json' % (self._intf._get_entry_point(), datatype))
    fields = get_column(search_fds, 'FIELD_ID', pattern)
    return ['%s/%s' % (datatype, field) if prepend_type else field for field in fields if '=' not in field and 'SHARINGSHAREPROJECT' not in field]