import copy
import sys
from functools import wraps
from types import FunctionType
import param
from . import util
from .pprint import PrettyPrinter
class Redim(metaclass=AccessorPipelineMeta):
    """
    Utility that supports re-dimensioning any HoloViews object via the
    redim method.
    """

    def __init__(self, obj, mode=None):
        self._obj = obj
        self.mode = mode

    def __str__(self):
        return '<holoviews.core.dimension.redim method>'

    @classmethod
    def replace_dimensions(cls, dimensions, overrides):
        """Replaces dimensions in list with dictionary of overrides.

        Args:
            dimensions: List of dimensions
            overrides: Dictionary of dimension specs indexed by name

        Returns:
            list: List of dimensions with replacements applied
        """
        from .dimension import Dimension
        replaced = []
        for d in dimensions:
            if d.name in overrides:
                override = overrides[d.name]
            elif d.label in overrides:
                override = overrides[d.label]
            else:
                override = None
            if override is None:
                replaced.append(d)
            elif isinstance(override, (str, tuple)):
                replaced.append(d.clone(override))
            elif isinstance(override, Dimension):
                replaced.append(override)
            elif isinstance(override, dict):
                replaced.append(d.clone(override.get('name', None), **{k: v for k, v in override.items() if k != 'name'}))
            else:
                raise ValueError('Dimension can only be overridden with another dimension or a dictionary of attributes')
        return replaced

    def _filter_cache(self, dmap, kdims):
        """
        Returns a filtered version of the DynamicMap cache leaving only
        keys consistently with the newly specified values
        """
        filtered = []
        for key, value in dmap.data.items():
            if not any((kd.values and v not in kd.values for kd, v in zip(kdims, key))):
                filtered.append((key, value))
        return filtered

    def _transform_dimension(self, kdims, vdims, dimension):
        if dimension in kdims:
            idx = kdims.index(dimension)
            dimension = self._obj.kdims[idx]
        elif dimension in vdims:
            idx = vdims.index(dimension)
            dimension = self._obj.vdims[idx]
        return dimension

    def _create_expression_transform(self, kdims, vdims, exclude=None):
        from ..util.transform import dim
        from .dimension import dimension_name
        if exclude is None:
            exclude = []

        def _transform_expression(expression):
            if dimension_name(expression.dimension) in exclude:
                dimension = expression.dimension
            else:
                dimension = self._transform_dimension(kdims, vdims, expression.dimension)
            expression = expression.clone(dimension)
            ops = []
            for op in expression.ops:
                new_op = dict(op)
                new_args = []
                for arg in op['args']:
                    if isinstance(arg, dim):
                        arg = _transform_expression(arg)
                    new_args.append(arg)
                new_op['args'] = tuple(new_args)
                new_kwargs = {}
                for kw, kwarg in op['kwargs'].items():
                    if isinstance(kwarg, dim):
                        kwarg = _transform_expression(kwarg)
                    new_kwargs[kw] = kwarg
                new_op['kwargs'] = new_kwargs
                ops.append(new_op)
            expression.ops = ops
            return expression
        return _transform_expression

    def __call__(self, specs=None, **dimensions):
        """
        Replace dimensions on the dataset and allows renaming
        dimensions in the dataset. Dimension mapping should map
        between the old dimension name and a dictionary of the new
        attributes, a completely new dimension or a new string name.
        """
        obj = self._obj
        redimmed = obj
        if obj._deep_indexable and self.mode != 'dataset':
            deep_mapped = [(k, v.redim(specs, **dimensions)) for k, v in obj.items()]
            redimmed = obj.clone(deep_mapped)
        if specs is not None:
            if not isinstance(specs, list):
                specs = [specs]
            matches = any((obj.matches(spec) for spec in specs))
            if self.mode != 'dynamic' and (not matches):
                return redimmed
        kdims = self.replace_dimensions(obj.kdims, dimensions)
        vdims = self.replace_dimensions(obj.vdims, dimensions)
        zipped_dims = zip(obj.kdims + obj.vdims, kdims + vdims)
        renames = {pk.name: nk for pk, nk in zipped_dims if pk.name != nk.name}
        if self.mode == 'dataset':
            data = obj.data
            if renames:
                data = obj.interface.redim(obj, renames)
            transform = self._create_expression_transform(kdims, vdims, list(renames.values()))
            transforms = obj._transforms + [transform]
            clone = obj.clone(data, kdims=kdims, vdims=vdims, transforms=transforms)
            if self._obj.dimensions(label='name') == clone.dimensions(label='name'):
                clone._plot_id = self._obj._plot_id
            return clone
        if self.mode != 'dynamic':
            return redimmed.clone(kdims=kdims, vdims=vdims)
        from ..util import Dynamic

        def dynamic_redim(obj, **dynkwargs):
            return obj.redim(specs, **dimensions)
        dmap = Dynamic(obj, streams=obj.streams, operation=dynamic_redim)
        dmap.data = dict(self._filter_cache(redimmed, kdims))
        with util.disable_constant(dmap):
            dmap.kdims = kdims
            dmap.vdims = vdims
        return dmap

    def _redim(self, name, specs, **dims):
        dimensions = {k: {name: v} for k, v in dims.items()}
        return self(specs, **dimensions)

    def cyclic(self, specs=None, **values):
        return self._redim('cyclic', specs, **values)

    def value_format(self, specs=None, **values):
        return self._redim('value_format', specs, **values)

    def range(self, specs=None, **values):
        return self._redim('range', specs, **values)

    def label(self, specs=None, **values):
        for k, v in values.items():
            dim = self._obj.get_dimension(k)
            if dim and dim.name != dim.label and (dim.label != v):
                raise ValueError('Cannot override an existing Dimension label')
        return self._redim('label', specs, **values)

    def soft_range(self, specs=None, **values):
        return self._redim('soft_range', specs, **values)

    def type(self, specs=None, **values):
        return self._redim('type', specs, **values)

    def nodata(self, specs=None, **values):
        return self._redim('nodata', specs, **values)

    def step(self, specs=None, **values):
        return self._redim('step', specs, **values)

    def default(self, specs=None, **values):
        return self._redim('default', specs, **values)

    def unit(self, specs=None, **values):
        return self._redim('unit', specs, **values)

    def values(self, specs=None, **ranges):
        return self._redim('values', specs, **ranges)