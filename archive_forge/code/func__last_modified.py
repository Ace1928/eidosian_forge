import re
from lxml import etree
from .jsonutil import JsonTable
def _last_modified(self):
    entry_point = self._intf._get_entry_point()
    uri = '%s/subjects?columns=last_modified' % entry_point
    return dict(JsonTable(self._intf._get_json(uri), order_by=['ID', 'last_modified']).select(['ID', 'last_modified']).items())