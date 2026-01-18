from copy import deepcopy
class IndexColumns(Columns):

    def __init__(self, table, columns, quote_name, col_suffixes=(), opclasses=()):
        self.opclasses = opclasses
        super().__init__(table, columns, quote_name, col_suffixes)

    def __str__(self):

        def col_str(column, idx):
            col = '{} {}'.format(self.quote_name(column), self.opclasses[idx])
            try:
                suffix = self.col_suffixes[idx]
                if suffix:
                    col = '{} {}'.format(col, suffix)
            except IndexError:
                pass
            return col
        return ', '.join((col_str(column, idx) for idx, column in enumerate(self.columns)))