from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def _make_data_table(self) -> DataTable:
    """ Builds the datatable portion of the final plot.

        """
    columns = [TableColumn(field='props', title='Property'), TableColumn(field='values', title='Value')]
    prop_source = ColumnDataSource(self._prop_df)
    model_id = self._node_source.data['index'][0]
    groupfilter = GroupFilter(column_name='id', group=model_id)
    data_table2_view = CDSView(filter=groupfilter)
    data_table2 = DataTable(source=prop_source, view=data_table2_view, columns=columns, visible=False, index_position=None, fit_columns=True, editable=False)
    self._groupfilter = groupfilter
    self._prop_source = prop_source
    return data_table2