from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
class UniformNdMapping(NdMapping):
    """
    A UniformNdMapping is a map of Dimensioned objects and is itself
    indexed over a number of specified dimensions. The dimension may
    be a spatial dimension (i.e., a ZStack), time (specifying a frame
    sequence) or any other combination of Dimensions.

    UniformNdMapping objects can be sliced, sampled, reduced, overlaid
    and split along its and its containing Element's dimensions.
    Subclasses should implement the appropriate slicing, sampling and
    reduction methods for their Dimensioned type.
    """
    data_type = (ViewableElement, NdMapping)
    __abstract = True
    _deep_indexable = True
    _auxiliary_component = False

    def __init__(self, initial_items=None, kdims=None, group=None, label=None, **params):
        self._type = None
        self._group_check, self.group = (None, group)
        self._label_check, self.label = (None, label)
        super().__init__(initial_items, kdims=kdims, **params)

    def clone(self, data=None, shared_data=True, new_type=None, link=True, *args, **overrides):
        """Clones the object, overriding data and parameters.

        Args:
            data: New data replacing the existing data
            shared_data (bool, optional): Whether to use existing data
            new_type (optional): Type to cast object to
            link (bool, optional): Whether clone should be linked
                Determines whether Streams and Links attached to
                original object will be inherited.
            *args: Additional arguments to pass to constructor
            **overrides: New keyword arguments to pass to constructor

        Returns:
            Cloned object
        """
        settings = self.param.values()
        if settings.get('group', None) != self._group:
            settings.pop('group')
        if settings.get('label', None) != self._label:
            settings.pop('label')
        if new_type is None:
            clone_type = self.__class__
        else:
            clone_type = new_type
            new_params = new_type.param.objects()
            settings = {k: v for k, v in settings.items() if k in new_params}
        settings = dict(settings, **overrides)
        if 'id' not in settings and new_type in [type(self), None]:
            settings['id'] = self.id
        if data is None and shared_data:
            data = self.data
            if link:
                settings['plot_id'] = self._plot_id
        pos_args = getattr(self, '_' + type(self).__name__ + '__pos_params', [])
        with item_check(not shared_data and self._check_items):
            return clone_type(data, *args, **{k: v for k, v in settings.items() if k not in pos_args})

    def collapse(self, dimensions=None, function=None, spreadfn=None, **kwargs):
        """Concatenates and aggregates along supplied dimensions

        Useful to collapse stacks of objects into a single object,
        e.g. to average a stack of Images or Curves.

        Args:
            dimensions: Dimension(s) to collapse
                Defaults to all key dimensions
            function: Aggregation function to apply, e.g. numpy.mean
            spreadfn: Secondary reduction to compute value spread
                Useful for computing a confidence interval, spread, or
                standard deviation.
            **kwargs: Keyword arguments passed to the aggregation function

        Returns:
            Returns the collapsed element or HoloMap of collapsed
            elements
        """
        from .data import concat
        from .overlay import CompositeOverlay
        if not dimensions:
            dimensions = self.kdims
        if not isinstance(dimensions, list):
            dimensions = [dimensions]
        if self.ndims > 1 and len(dimensions) != self.ndims:
            groups = self.groupby([dim for dim in self.kdims if dim not in dimensions])
        elif all((d in self.kdims for d in dimensions)):
            groups = UniformNdMapping([(0, self)], ['tmp'])
        else:
            raise KeyError('Supplied dimensions not found.')
        collapsed = groups.clone(shared_data=False)
        for key, group in groups.items():
            last = group.values()[-1]
            if isinstance(last, UniformNdMapping):
                group_data = dict([(k, v.collapse()) for k, v in group.items()])
                group = group.clone(group_data)
            if hasattr(group.values()[-1], 'interface'):
                group_data = concat(group)
                if function:
                    agg = group_data.aggregate(group.last.kdims, function, spreadfn, **kwargs)
                    group_data = group.type(agg)
            elif issubclass(group.type, CompositeOverlay) and hasattr(self, '_split_overlays'):
                keys, maps = self._split_overlays()
                group_data = group.type(dict([(key, ndmap.collapse(function=function, spreadfn=spreadfn, **kwargs)) for key, ndmap in zip(keys, maps)]))
            else:
                raise ValueError('Could not determine correct collapse operation for items of type: {group.type!r}.')
            collapsed[key] = group_data
        return collapsed if self.ndims - len(dimensions) else collapsed.last

    def dframe(self, dimensions=None, multi_index=False):
        """Convert dimension values to DataFrame.

        Returns a pandas dataframe of columns along each dimension,
        either completely flat or indexed by key dimensions.

        Args:
            dimensions: Dimensions to return as columns
            multi_index: Convert key dimensions to (multi-)index

        Returns:
            DataFrame of columns corresponding to each dimension
        """
        if dimensions is None:
            outer_dimensions = self.kdims
            inner_dimensions = None
        else:
            outer_dimensions = [self.get_dimension(d) for d in dimensions if d in self.kdims]
            inner_dimensions = [d for d in dimensions if d not in outer_dimensions]
        inds = [(d, self.get_dimension_index(d)) for d in outer_dimensions]
        dframes = []
        for key, element in self.data.items():
            df = element.dframe(inner_dimensions, multi_index)
            names = [d.name for d in outer_dimensions]
            key_dims = [(d.name, key[i]) for d, i in inds]
            if multi_index:
                length = len(df)
                indexes = [[v] * length for _, v in key_dims]
                if df.index.names != [None]:
                    indexes += [df.index]
                    names += list(df.index.names)
                df = df.set_index(indexes)
                df.index.names = names
            else:
                for dim, val in key_dims:
                    dimn = 1
                    while dim in df:
                        dim = dim + '_%d' % dimn
                        if dim in df:
                            dimn += 1
                    df.insert(0, dim, val)
            dframes.append(df)
        return pd.concat(dframes)

    @property
    def group(self):
        """Group inherited from items"""
        if self._group:
            return self._group
        group = get_ndmapping_label(self, 'group') if len(self) else None
        if group is None:
            return type(self).__name__
        return group

    @group.setter
    def group(self, group):
        if group is not None and (not sanitize_identifier.allowable(group)):
            raise ValueError('Supplied group %s contains invalid characters.' % self.group)
        self._group = group

    @property
    def label(self):
        """Label inherited from items"""
        if self._label:
            return self._label
        elif len(self):
            label = get_ndmapping_label(self, 'label')
            return '' if label is None else label
        else:
            return ''

    @label.setter
    def label(self, label):
        if label is not None and (not sanitize_identifier.allowable(label)):
            raise ValueError('Supplied group %s contains invalid characters.' % self.group)
        self._label = label

    @property
    def type(self):
        """The type of elements stored in the mapping."""
        if self._type is None and len(self):
            self._type = self.values()[0].__class__
        return self._type

    @property
    def empty_element(self):
        return self.type(None)

    def _item_check(self, dim_vals, data):
        if not self._check_items:
            return
        elif self.type is not None and type(data) != self.type:
            raise AssertionError(f'{self.__class__.__name__} must only contain one type of object, not both {type(data).__name__} and {self.type.__name__}.')
        super()._item_check(dim_vals, data)

    def __mul__(self, other, reverse=False):
        from .overlay import Overlay
        if isinstance(other, type(self)):
            if self.kdims != other.kdims:
                raise KeyError('Can only overlay two %ss with non-matching key dimensions.' % type(self).__name__)
            items = []
            self_keys = list(self.data.keys())
            other_keys = list(other.data.keys())
            for key in util.unique_iterator(self_keys + other_keys):
                self_el = self.data.get(key)
                other_el = other.data.get(key)
                if self_el is None:
                    item = [other_el]
                elif other_el is None:
                    item = [self_el]
                elif reverse:
                    item = [other_el, self_el]
                else:
                    item = [self_el, other_el]
                items.append((key, Overlay(item)))
            return self.clone(items)
        overlayed_items = [(k, other * el if reverse else el * other) for k, el in self.items()]
        try:
            return self.clone(overlayed_items)
        except NotImplementedError:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other, reverse=True)