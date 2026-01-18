import datetime
import functools
from cliff import columns as cliff_columns
from osc_lib.command import command
from osc_lib import utils
from openstackclient.i18n import _
class FloatColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return float('%.2f' % self._value)