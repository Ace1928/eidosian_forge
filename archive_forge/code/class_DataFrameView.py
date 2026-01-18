from __future__ import division, print_function, absolute_import
import inspect
from petl.util.base import Table
class DataFrameView(Table):

    def __init__(self, df, include_index=False):
        assert hasattr(df, 'columns') and hasattr(df, 'iterrows') and inspect.ismethod(df.iterrows), 'bad argument, expected pandas.DataFrame, found %r' % df
        self.df = df
        self.include_index = include_index

    def __iter__(self):
        if self.include_index:
            yield (('index',) + tuple(self.df.columns))
            for i, row in self.df.iterrows():
                yield ((i,) + tuple(row))
        else:
            yield tuple(self.df.columns)
            for _, row in self.df.iterrows():
                yield tuple(row)