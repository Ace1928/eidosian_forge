from cliff import columns
from osc_lib import utils
class ListDictColumn(columns.FormattableColumn):
    """Format column for list of dict content"""

    def human_readable(self):
        return utils.format_list_of_dicts(self._value)

    def machine_readable(self):
        return [dict(x) for x in self._value or []]