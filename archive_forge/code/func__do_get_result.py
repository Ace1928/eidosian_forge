import re
import warnings
from . import err
def _do_get_result(self):
    super()._do_get_result()
    fields = []
    if self.description:
        for f in self._result.fields:
            name = f.name
            if name in fields:
                name = f.table_name + '.' + name
            fields.append(name)
        self._fields = fields
    if fields and self._rows:
        self._rows = [self._conv_row(r) for r in self._rows]