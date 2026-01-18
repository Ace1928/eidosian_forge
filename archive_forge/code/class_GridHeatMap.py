import os
import json
from warnings import warn
import ipywidgets as widgets
from ipywidgets import (Widget, DOMWidget, CallbackDispatcher,
from traitlets import (Int, Unicode, List, Enum, Dict, Bool, Float,
from traittypes import Array
from numpy import histogram
import numpy as np
from .scales import Scale, OrdinalScale, LinearScale
from .traits import (Date, array_serialization,
from ._version import __frontend_version__
from .colorschemes import CATEGORY10
@register_mark('bqplot.GridHeatMap')
class GridHeatMap(Mark):
    """GridHeatMap mark.

    Alignment: The tiles can be aligned so that the data matches either the
    start, the end or the midpoints of the tiles. This is controlled by the
    align attribute.

    Suppose the data passed is a m-by-n matrix. If the scale for the rows is
    Ordinal, then alignment is by default the mid points. For a non-ordinal
    scale, the data cannot be aligned to the mid points of the rectangles.

    If it is not ordinal, then two cases arise. If the number of rows passed
    is m, then align attribute can be used. If the number of rows passed
    is m+1, then the data are the boundaries of the m rectangles.

    If rows and columns are not passed, and scales for them are also
    not passed, then ordinal scales are generated for the rows and columns.

    Attributes
    ----------
    row_align: Enum(['start', 'end'])
        This is only valid if the number of entries in `row` exactly match the
        number of rows in `color` and the `row_scale` is not `OrdinalScale`.
        `start` aligns the row values passed to be aligned with the start
        of the tiles and `end` aligns the row values to the end of the tiles.
    column_align: Enum(['start', end'])
        This is only valid if the number of entries in `column` exactly
        match the number of columns in `color` and the `column_scale` is
        not `OrdinalScale`. `start` aligns the column values passed to
        be aligned with the start of the tiles and `end` aligns the
        column values to the end of the tiles.
    anchor_style: dict (default: {})
        Controls the style for the element which serves as the anchor during
        selection.
    display_format: string (default: None)
        format for displaying values. If None, then values are not displayed
    font_style: dict
        CSS style for the text of each cell

    Data Attributes

    color: numpy.ndarray or None (default: None)
        color of the data points (2d array). The number of elements in
        this array correspond to the number of cells created in the heatmap.
    row: numpy.ndarray or None (default: None)
        labels for the rows of the `color` array passed. The length of
        this can be no more than 1 away from the number of rows in `color`.
        This is a scaled attribute and can be used to affect the height of the
        cells as the entries of `row` can indicate the start or the end points
        of the cells. Refer to the property `row_align`.
        If this property is None, then a uniformly spaced grid is generated in
        the row direction.
    column: numpy.ndarray or None (default: None)
        labels for the columns of the `color` array passed. The length of
        this can be no more than 1 away from the number of columns in `color`
        This is a scaled attribute and can be used to affect the width of the
        cells as the entries of `column` can indicate the start or the
        end points of the cells. Refer to the property `column_align`.
        If this property is None, then a uniformly spaced grid is generated in
        the column direction.
    """
    row = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    column = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Number', atype='bqplot.Axis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 1))
    color = Array(None, allow_none=True).tag(sync=True, scaled=True, rtype='Color', atype='bqplot.ColorAxis', **array_serialization).valid(array_squeeze, array_dimension_bounds(1, 2))
    scales_metadata = Dict({'row': {'orientation': 'vertical', 'dimension': 'y'}, 'column': {'orientation': 'horizontal', 'dimension': 'x'}, 'color': {'dimension': 'color'}}).tag(sync=True)
    row_align = Enum(['start', 'end'], default_value='start').tag(sync=True)
    column_align = Enum(['start', 'end'], default_value='start').tag(sync=True)
    null_color = Color('black', allow_none=True).tag(sync=True)
    stroke = Color('black', allow_none=True).tag(sync=True)
    opacity = Float(1.0, min=0.2, max=1).tag(sync=True, display_name='Opacity')
    anchor_style = Dict().tag(sync=True)
    display_format = Unicode(default_value=None, allow_none=True).tag(sync=True)
    font_style = Dict().tag(sync=True)

    def __init__(self, **kwargs):
        scales = kwargs.pop('scales', {})
        if scales.get('row', None) is None:
            row_scale = OrdinalScale(reverse=True)
            scales['row'] = row_scale
        if scales.get('column', None) is None:
            column_scale = OrdinalScale()
            scales['column'] = column_scale
        kwargs['scales'] = scales
        super(GridHeatMap, self).__init__(**kwargs)

    @validate('row')
    def _validate_row(self, proposal):
        row = proposal.value
        if row is None:
            return row
        color = np.asarray(self.color)
        n_rows = color.shape[0]
        if len(row) != n_rows and len(row) != n_rows + 1 and (len(row) != n_rows - 1):
            raise TraitError('row must be an array of size color.shape[0]')
        return row

    @validate('column')
    def _validate_column(self, proposal):
        column = proposal.value
        if column is None:
            return column
        color = np.asarray(self.color)
        n_columns = color.shape[1]
        if len(column) != n_columns and len(column) != n_columns + 1 and (len(column) != n_columns - 1):
            raise TraitError('column must be an array of size color.shape[1]')
        return column
    _view_name = Unicode('GridHeatMap').tag(sync=True)
    _model_name = Unicode('GridHeatMapModel').tag(sync=True)