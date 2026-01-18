from cliff import columns as cliff_columns
class AdminStateColumn(cliff_columns.FormattableColumn):

    def human_readable(self):
        return 'UP' if self._value else 'DOWN'