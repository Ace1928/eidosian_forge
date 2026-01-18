from cliff import columns
from osc_lib import utils
class SizeColumn(columns.FormattableColumn):
    """Format column for file size content"""

    def human_readable(self):
        return utils.format_size(self._value)