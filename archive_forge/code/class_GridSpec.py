from __future__ import annotations
import math
from collections import namedtuple
from functools import partial
from typing import (
import numpy as np
import param
from bokeh.models import FlexBox as BkFlexBox, GridBox as BkGridBox
from ..io.document import freeze_doc
from ..io.model import hold
from ..io.resources import CDN_DIST
from .base import (
class GridSpec(Panel):
    """
    The `GridSpec` is an *array like* layout that allows arranging multiple Panel
    objects in a grid using a simple API to assign objects to individual grid cells or
    to a grid span.

    Other layout containers function like lists, but a GridSpec has an API similar
    to a 2D array, making it possible to use 2D assignment to populate, index, and slice
    the grid.

    See `GridStack` for a similar layout that allows the user to resize and drag the
    cells.

    Reference: https://panel.holoviz.org/reference/layouts/GridSpec.html

    :Example:

    >>> import panel as pn
    >>> gspec = pn.GridSpec(width=800, height=600)
    >>> gspec[:,   0  ] = pn.Spacer(styles=dict(background='red'))
    >>> gspec[0,   1:3] = pn.Spacer(styles=dict(background='green'))
    >>> gspec[1,   2:4] = pn.Spacer(styles=dict(background='orange'))
    >>> gspec[2,   1:4] = pn.Spacer(styles=dict(background='blue'))
    >>> gspec[0:1, 3:4] = pn.Spacer(styles=dict(background='purple'))
    >>> gspec
    """
    objects = param.Dict(default={}, doc='\n        The dictionary of child objects that make up the grid.')
    mode = param.ObjectSelector(default='warn', objects=['warn', 'error', 'override'], doc='\n        Whether to warn, error or simply override on overlapping assignment.')
    ncols = param.Integer(default=None, bounds=(0, None), doc='\n        Limits the number of columns that can be assigned.')
    nrows = param.Integer(default=None, bounds=(0, None), doc='\n        Limits the number of rows that can be assigned.')
    _bokeh_model: ClassVar[Model] = BkGridBox
    _linked_properties: ClassVar[Tuple[str]] = ()
    _rename: ClassVar[Mapping[str, str | None]] = {'objects': 'children', 'mode': None, 'ncols': None, 'nrows': None}
    _source_transforms: ClassVar[Mapping[str, str | None]] = {'objects': None, 'mode': None}
    _preprocess_params: ClassVar[List[str]] = ['objects']
    _stylesheets: ClassVar[List[str]] = [f'{CDN_DIST}css/gridspec.css']

    def __init__(self, **params):
        if 'objects' not in params:
            params['objects'] = {}
        super().__init__(**params)
        self._updating = False
        self._update_nrows()
        self._update_ncols()
        self._update_grid_size()

    @param.depends('nrows', watch=True)
    def _update_nrows(self):
        if not self._updating:
            self._rows_fixed = bool(self.nrows)

    @param.depends('ncols', watch=True)
    def _update_ncols(self):
        if not self._updating:
            self._cols_fixed = self.ncols is not None

    @param.depends('objects', watch=True)
    def _update_grid_size(self):
        self._updating = True
        if not self._cols_fixed:
            max_xidx = [x1 for _, _, _, x1 in self.objects if x1 is not None]
            self.ncols = max(max_xidx) if max_xidx else 1 if len(self.objects) else 0
        if not self._rows_fixed:
            max_yidx = [y1 for _, _, y1, _ in self.objects if y1 is not None]
            self.nrows = max(max_yidx) if max_yidx else 1 if len(self.objects) else 0
        self._updating = False

    def _init_params(self):
        params = super()._init_params()
        if self.sizing_mode not in ['fixed', None]:
            if 'min_width' not in params and 'width' in params:
                params['min_width'] = params['width']
            if 'min_height' not in params and 'height' in params:
                params['min_height'] = params['height']
        return params

    def _get_objects(self, model, old_objects, doc, root, comm=None):
        from ..pane.base import RerenderError
        if self.ncols and self.width:
            width = self.width / self.ncols
        else:
            width = 0
        if self.nrows and self.height:
            height = self.height / self.nrows
        else:
            height = 0
        current_objects = list(self.objects.values())
        if isinstance(old_objects, dict):
            old_objects = list(old_objects.values())
        for old in old_objects:
            if old not in current_objects:
                old._cleanup(root)
        children, old_children = ([], [])
        for i, ((y0, x0, y1, x1), obj) in enumerate(self.objects.items()):
            x0 = 0 if x0 is None else x0
            x1 = self.ncols if x1 is None else x1
            y0 = 0 if y0 is None else y0
            y1 = self.nrows if y1 is None else y1
            r, c, h, w = (y0, x0, y1 - y0, x1 - x0)
            properties = {}
            if self.sizing_mode in ['fixed', None]:
                if width:
                    properties['width'] = int(w * width)
                if height:
                    properties['height'] = int(h * height)
            else:
                properties['sizing_mode'] = self.sizing_mode
                if 'width' in self.sizing_mode and height:
                    properties['height'] = int(h * height)
                elif 'height' in self.sizing_mode and width:
                    properties['width'] = int(w * width)
            obj.param.update(**{k: v for k, v in properties.items() if not obj.param[k].readonly})
            if obj in old_objects:
                child, _ = obj._models[root.ref['id']]
                old_children.append(child)
            else:
                try:
                    child = obj._get_model(doc, root, model, comm)
                except RerenderError as e:
                    if e.layout is not None and e.layout is not self:
                        raise e
                    e.layout = None
                    return self._get_objects(model, current_objects[:i], doc, root, comm)
            if isinstance(child, BkFlexBox) and len(child.children) == 1:
                child.children[0].update(**properties)
            else:
                child.update(**properties)
            children.append((child, r, c, h, w))
        return (children, old_children)

    def _compute_sizing_mode(self, children, props):
        children = [child for child, _, _, _, _ in children]
        return super()._compute_sizing_mode(children, props)

    @property
    def _xoffset(self):
        min_xidx = [x0 for _, x0, _, _ in self.objects if x0 is not None]
        return min(min_xidx) if min_xidx and len(min_xidx) == len(self.objects) else 0

    @property
    def _yoffset(self):
        min_yidx = [y0 for y0, x0, _, _ in self.objects if y0 is not None]
        return min(min_yidx) if min_yidx and len(min_yidx) == len(self.objects) else 0

    @property
    def _object_grid(self):
        grid = np.full((self.nrows, self.ncols), None, dtype=object)
        for (y0, x0, y1, x1), obj in self.objects.items():
            l = 0 if x0 is None else x0
            r = self.ncols if x1 is None else x1
            t = 0 if y0 is None else y0
            b = self.nrows if y1 is None else y1
            for y in range(t, b):
                for x in range(l, r):
                    grid[y, x] = {((y0, x0, y1, x1), obj)}
        return grid

    def _cleanup(self, root: Model | None=None) -> None:
        super()._cleanup(root)
        for p in self.objects.values():
            p._cleanup(root)

    @property
    def grid(self):
        grid = np.zeros((self.nrows, self.ncols), dtype='uint8')
        for y0, x0, y1, x1 in self.objects:
            grid[y0:y1, x0:x1] += 1
        return grid

    def clone(self, **params):
        """
        Makes a copy of the GridSpec sharing the same parameters.

        Arguments
        ---------
        params: Keyword arguments override the parameters on the clone.

        Returns
        -------
        Cloned GridSpec object
        """
        p = dict(self.param.values(), **params)
        if not self._cols_fixed:
            del p['ncols']
        if not self._rows_fixed:
            del p['nrows']
        return type(self)(**p)

    def __iter__(self):
        for obj in self.objects.values():
            yield obj

    def __delitem__(self, index):
        if isinstance(index, tuple):
            yidx, xidx = index
        else:
            yidx, xidx = (index, slice(None))
        subgrid = self._object_grid[yidx, xidx]
        if isinstance(subgrid, np.ndarray):
            deleted = dict([list(o)[0] for o in subgrid.flatten()])
        else:
            deleted = [list(subgrid)[0][0]]
        for key in deleted:
            del self.objects[key]
        self.param.trigger('objects')

    def __getitem__(self, index):
        if isinstance(index, tuple):
            yidx, xidx = index
        else:
            yidx, xidx = (index, slice(None))
        subgrid = self._object_grid[yidx, xidx]
        if isinstance(subgrid, np.ndarray):
            objects = dict([list(o)[0] for o in subgrid.flatten()])
            gspec = self.clone(objects=objects)
            xoff, yoff = (gspec._xoffset, gspec._yoffset)
            adjusted = []
            for (y0, x0, y1, x1), obj in gspec.objects.items():
                if y0 is not None:
                    y0 -= yoff
                if y1 is not None:
                    y1 -= yoff
                if x0 is not None:
                    x0 -= xoff
                if x1 is not None:
                    x1 -= xoff
                if ((y0, x0, y1, x1), obj) not in adjusted:
                    adjusted.append(((y0, x0, y1, x1), obj))
            gspec.objects = dict(adjusted)
            width_scale = gspec.ncols / float(self.ncols)
            height_scale = gspec.nrows / float(self.nrows)
            if gspec.width:
                gspec.width = int(gspec.width * width_scale)
            if gspec.height:
                gspec.height = int(gspec.height * height_scale)
            if gspec.max_width:
                gspec.max_width = int(gspec.max_width * width_scale)
            if gspec.max_height:
                gspec.max_height = int(gspec.max_height * height_scale)
            return gspec
        else:
            return list(subgrid)[0][1]

    def __setitem__(self, index, obj):
        from ..pane.base import panel
        if not isinstance(index, tuple):
            raise IndexError('Must supply a 2D index for GridSpec assignment.')
        yidx, xidx = index
        if isinstance(xidx, slice):
            x0, x1 = (xidx.start, xidx.stop)
        else:
            x0, x1 = (xidx, xidx + 1)
        if isinstance(yidx, slice):
            y0, y1 = (yidx.start, yidx.stop)
        else:
            y0, y1 = (yidx, yidx + 1)
        l = 0 if x0 is None else x0
        r = self.ncols if x1 is None else x1
        t = 0 if y0 is None else y0
        b = self.nrows if y1 is None else y1
        if self._cols_fixed and (l >= self.ncols or r > self.ncols):
            raise IndexError(f'Assigned object to column(s) out-of-bounds of the grid declared by `ncols`. which was set to {self.ncols}.')
        if self._rows_fixed and (t >= self.nrows or b > self.nrows):
            raise IndexError(f'Assigned object to column(s) out-of-bounds of the grid declared by `nrows`, which was set to {self.nrows}.')
        key = (y0, x0, y1, x1)
        overlap = key in self.objects
        clone = self.clone(objects=dict(self.objects), mode='override')
        if not overlap:
            clone.objects[key] = panel(obj)
            clone._update_grid_size()
            grid = clone.grid
        else:
            grid = clone.grid
            grid[t:b, l:r] += 1
        overlap_grid = grid > 1
        new_objects = dict(self.objects)
        if overlap_grid.any():
            overlapping = ''
            objects = []
            for yidx, xidx in zip(*np.where(overlap_grid)):
                try:
                    old_obj = self[yidx, xidx]
                except Exception:
                    continue
                if old_obj not in objects:
                    objects.append(old_obj)
                    overlapping += '    (%d, %d): %s\n\n' % (yidx, xidx, old_obj)
            overlap_text = 'Specified region overlaps with the following existing object(s) in the grid:\n\n' + overlapping + 'The following shows a view of the grid (empty: 0, occupied: 1, overlapping: 2):\n\n' + str(grid.astype('uint8'))
            if self.mode == 'error':
                raise IndexError(overlap_text)
            elif self.mode == 'warn':
                self.param.warning(overlap_text)
            subgrid = self._object_grid[index]
            if isinstance(subgrid, set):
                objects = [list(subgrid)[0][0]] if subgrid else []
            else:
                objects = [list(o)[0][0] for o in subgrid.flatten()]
            for dkey in set(objects):
                try:
                    del new_objects[dkey]
                except KeyError:
                    continue
        new_objects[key] = panel(obj)
        self.objects = new_objects