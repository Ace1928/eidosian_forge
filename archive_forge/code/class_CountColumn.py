import datetime
import functools
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class CountColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return len(self._value) if self._value is not None else None