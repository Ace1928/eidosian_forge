import testtools
from barbicanclient import formatter
class EntityFormatter(formatter.EntityFormatter):
    columns = ('Column A', 'Column B', 'Column C')

    def _get_formatted_data(self):
        data = (self._attr_a, self._attr_b, self._attr_c)
        return data