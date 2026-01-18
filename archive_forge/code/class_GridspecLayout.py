import re
from collections import defaultdict
from traitlets import Instance, Bool, Unicode, CUnicode, CaselessStrEnum, Tuple
from traitlets import Integer
from traitlets import HasTraits, TraitError
from traitlets import observe, validate
from .widget import Widget
from .widget_box import GridBox
from .docutils import doc_subst
@doc_subst(_doc_snippets)
class GridspecLayout(GridBox, LayoutProperties):
    """ Define a N by M grid layout

    Parameters
    ----------

    n_rows : int
        number of rows in the grid

    n_columns : int
        number of columns in the grid

    {style_params}

    Examples
    --------

    >>> from ipywidgets import GridspecLayout, Button, Layout
    >>> layout = GridspecLayout(n_rows=4, n_columns=2, height='200px')
    >>> layout[:3, 0] = Button(layout=Layout(height='auto', width='auto'))
    >>> layout[1:, 1] = Button(layout=Layout(height='auto', width='auto'))
    >>> layout[-1, 0] = Button(layout=Layout(height='auto', width='auto'))
    >>> layout[0, 1] = Button(layout=Layout(height='auto', width='auto'))
    >>> layout
    """
    n_rows = Integer()
    n_columns = Integer()

    def __init__(self, n_rows=None, n_columns=None, **kwargs):
        super().__init__(**kwargs)
        self.n_rows = n_rows
        self.n_columns = n_columns
        self._grid_template_areas = [['.'] * self.n_columns for i in range(self.n_rows)]
        self._grid_template_rows = 'repeat(%d, 1fr)' % (self.n_rows,)
        self._grid_template_columns = 'repeat(%d, 1fr)' % (self.n_columns,)
        self._children = {}
        self._id_count = 0

    @validate('n_rows', 'n_columns')
    def _validate_integer(self, proposal):
        if proposal['value'] > 0:
            return proposal['value']
        raise TraitError('n_rows and n_columns must be positive integer')

    def _get_indices_from_slice(self, row, column):
        """convert a two-dimensional slice to a list of rows and column indices"""
        if isinstance(row, slice):
            start, stop, stride = row.indices(self.n_rows)
            rows = range(start, stop, stride)
        else:
            rows = [row]
        if isinstance(column, slice):
            start, stop, stride = column.indices(self.n_columns)
            columns = range(start, stop, stride)
        else:
            columns = [column]
        return (rows, columns)

    def __setitem__(self, key, value):
        row, column = key
        self._id_count += 1
        obj_id = 'widget%03d' % self._id_count
        value.layout.grid_area = obj_id
        rows, columns = self._get_indices_from_slice(row, column)
        for row in rows:
            for column in columns:
                current_value = self._grid_template_areas[row][column]
                if current_value != '.' and current_value in self._children:
                    del self._children[current_value]
                self._grid_template_areas[row][column] = obj_id
        self._children[obj_id] = value
        self._update_layout()

    def __getitem__(self, key):
        rows, columns = self._get_indices_from_slice(*key)
        obj_id = None
        for row in rows:
            for column in columns:
                new_obj_id = self._grid_template_areas[row][column]
                obj_id = obj_id or new_obj_id
                if obj_id != new_obj_id:
                    raise TypeError('The slice spans several widgets, but only a single widget can be retrieved at a time')
        return self._children[obj_id]

    def _update_layout(self):
        grid_template_areas_css = '\n'.join(('"{}"'.format(' '.join(line)) for line in self._grid_template_areas))
        self.layout.grid_template_columns = self._grid_template_columns
        self.layout.grid_template_rows = self._grid_template_rows
        self.layout.grid_template_areas = grid_template_areas_css
        self.children = tuple(self._children.values())