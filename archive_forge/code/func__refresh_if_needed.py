import collections
import weakref
from heat.common import context
from heat.common import exception
from heat.common.i18n import _
from heat.db import api as db_api
from heat.objects import raw_template_files
def _refresh_if_needed(self):
    if self.files_id is None:
        return
    if self.files_id in _d:
        self.files = _d[self.files_id]
        return
    self._refresh()