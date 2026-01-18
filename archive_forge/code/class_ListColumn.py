from cliff import columns
from osc_lib import utils
class ListColumn(columns.FormattableColumn):
    """Format column for list content"""

    def human_readable(self):
        return utils.format_list(self._value)

    def machine_readable(self):
        return [x for x in self._value or []]