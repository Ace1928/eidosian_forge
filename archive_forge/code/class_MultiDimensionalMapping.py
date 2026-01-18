from itertools import cycle
from operator import itemgetter
import numpy as np
import pandas as pd
import param
from . import util
from .dimension import Dimension, Dimensioned, ViewableElement, asdim
from .util import (
class MultiDimensionalMapping(Dimensioned):
    """
    An MultiDimensionalMapping is a Dimensioned mapping (like a
    dictionary or array) that uses fixed-length multidimensional
    keys. This behaves like a sparse N-dimensional array that does not
    require a dense sampling over the multidimensional space.

    If the underlying value for each (key, value) pair also supports
    indexing (such as a dictionary, array, or list), fully qualified
    (deep) indexing may be used from the top level, with the first N
    dimensions of the index selecting a particular Dimensioned object
    and the remaining dimensions indexing into that object.

    For instance, for a MultiDimensionalMapping with dimensions "Year"
    and "Month" and underlying values that are 2D floating-point
    arrays indexed by (r,c), a 2D array may be indexed with x[2000,3]
    and a single floating-point number may be indexed as
    x[2000,3,1,9].

    In practice, this class is typically only used as an abstract base
    class, because the NdMapping subclass extends it with a range of
    useful slicing methods for selecting subsets of the data. Even so,
    keeping the slicing support separate from the indexing and data
    storage methods helps make both classes easier to understand.
    """
    group = param.String(default='MultiDimensionalMapping', constant=True)
    kdims = param.List(default=[Dimension('Default')], constant=True)
    vdims = param.List(default=[], bounds=(0, 0), constant=True)
    sort = param.Boolean(default=True, doc='\n        Whether the items should be sorted in the constructor.')
    data_type = None
    _deep_indexable = False
    _check_items = True

    def __init__(self, initial_items=None, kdims=None, **params):
        if isinstance(initial_items, MultiDimensionalMapping):
            params = dict(util.get_param_values(initial_items), **dict(params))
        if kdims is not None:
            params['kdims'] = kdims
        super().__init__({}, **dict(params))
        self._next_ind = 0
        self._check_key_type = True
        if initial_items is None:
            initial_items = []
        if isinstance(initial_items, tuple):
            self._add_item(initial_items[0], initial_items[1])
        elif not self._check_items:
            if isinstance(initial_items, dict):
                initial_items = initial_items.items()
            elif isinstance(initial_items, MultiDimensionalMapping):
                initial_items = initial_items.data.items()
            self.data = dict(((k if isinstance(k, tuple) else (k,), v) for k, v in initial_items))
            if self.sort:
                self._resort()
        elif initial_items is not None:
            self.update(dict(initial_items))

    def _item_check(self, dim_vals, data):
        """
        Applies optional checks to individual data elements before
        they are inserted ensuring that they are of a certain
        type. Subclassed may implement further element restrictions.
        """
        if not self._check_items:
            return
        elif self.data_type is not None and (not isinstance(data, self.data_type)):
            if isinstance(self.data_type, tuple):
                data_type = tuple((dt.__name__ for dt in self.data_type))
            else:
                data_type = self.data_type.__name__
            slf = type(self).__name__
            data = type(data).__name__
            raise TypeError(f'{slf} does not accept {data} type, data elements have to be a {data_type}.')
        elif not len(dim_vals) == self.ndims:
            raise KeyError('The data contains keys of length %d, but the kdims only declare %d dimensions. Ensure that the number of kdims match the length of the keys in your data.' % (len(dim_vals), self.ndims))

    def _add_item(self, dim_vals, data, sort=True, update=True):
        """
        Adds item to the data, applying dimension types and ensuring
        key conforms to Dimension type and values.
        """
        sort = sort and self.sort
        if not isinstance(dim_vals, tuple):
            dim_vals = (dim_vals,)
        self._item_check(dim_vals, data)
        dim_types = zip([kd.type for kd in self.kdims], dim_vals)
        dim_vals = tuple((v if None in [t, v] else t(v) for t, v in dim_types))
        valid_vals = zip(self.kdims, dim_vals)
        for dim, val in valid_vals:
            if dim.values and val is not None and (val not in dim.values):
                raise KeyError(f'{dim} dimension value {val!r} not in specified dimension values.')
        if update and dim_vals in self.data and isinstance(self.data[dim_vals], (MultiDimensionalMapping, dict)):
            self.data[dim_vals].update(data)
        else:
            self.data[dim_vals] = data
        if sort:
            self._resort()

    def _apply_key_type(self, keys):
        """
        If a type is specified by the corresponding key dimension,
        this method applies the type to the supplied key.
        """
        typed_key = ()
        for dim, key in zip(self.kdims, keys):
            key_type = dim.type
            if key_type is None:
                typed_key += (key,)
            elif isinstance(key, slice):
                sl_vals = [key.start, key.stop, key.step]
                typed_key += (slice(*[key_type(el) if el is not None else None for el in sl_vals]),)
            elif key is Ellipsis:
                typed_key += (key,)
            elif isinstance(key, list):
                typed_key += ([key_type(k) for k in key],)
            else:
                typed_key += (key_type(key),)
        return typed_key

    def _split_index(self, key):
        """
        Partitions key into key and deep dimension groups. If only key
        indices are supplied, the data is indexed with an empty tuple.
        Keys with indices than there are dimensions will be padded.
        """
        if not isinstance(key, tuple):
            key = (key,)
        elif key == ():
            return ((), ())
        if key[0] is Ellipsis:
            num_pad = self.ndims - len(key) + 1
            key = (slice(None),) * num_pad + key[1:]
        elif len(key) < self.ndims:
            num_pad = self.ndims - len(key)
            key = key + (slice(None),) * num_pad
        map_slice = key[:self.ndims]
        if self._check_key_type:
            map_slice = self._apply_key_type(map_slice)
        if len(key) == self.ndims:
            return (map_slice, ())
        else:
            return (map_slice, key[self.ndims:])

    def _dataslice(self, data, indices):
        """
        Returns slice of data element if the item is deep
        indexable. Warns if attempting to slice an object that has not
        been declared deep indexable.
        """
        if self._deep_indexable and isinstance(data, Dimensioned) and indices:
            return data[indices]
        elif len(indices) > 0:
            self.param.warning('Cannot index into data element, extra data indices ignored.')
        return data

    def _resort(self):
        self.data = dict(dimension_sort(self.data, self.kdims, self.vdims, range(self.ndims)))

    def clone(self, data=None, shared_data=True, *args, **overrides):
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
        with item_check(not shared_data and self._check_items):
            return super().clone(data, shared_data, *args, **overrides)

    def groupby(self, dimensions, container_type=None, group_type=None, **kwargs):
        """Groups object by one or more dimensions

        Applies groupby operation over the specified dimensions
        returning an object of type container_type (expected to be
        dictionary-like) containing the groups.

        Args:
            dimensions: Dimension(s) to group by
            container_type: Type to cast group container to
            group_type: Type to cast each group to
            dynamic: Whether to return a DynamicMap
            **kwargs: Keyword arguments to pass to each group

        Returns:
            Returns object of supplied container_type containing the
            groups. If dynamic=True returns a DynamicMap instead.
        """
        if self.ndims == 1:
            self.param.warning('Cannot split Map with only one dimension.')
            return self
        elif not isinstance(dimensions, list):
            dimensions = [dimensions]
        container_type = container_type if container_type else type(self)
        group_type = group_type if group_type else type(self)
        dimensions = [self.get_dimension(d, strict=True) for d in dimensions]
        with item_check(False):
            sort = kwargs.pop('sort', self.sort)
            return util.ndmapping_groupby(self, dimensions, container_type, group_type, sort=sort, **kwargs)

    def add_dimension(self, dimension, dim_pos, dim_val, vdim=False, **kwargs):
        """Adds a dimension and its values to the object

        Requires the dimension name or object, the desired position in
        the key dimensions and a key value scalar or sequence of the
        same length as the existing keys.

        Args:
            dimension: Dimension or dimension spec to add
            dim_pos (int) Integer index to insert dimension at
            dim_val (scalar or ndarray): Dimension value(s) to add
            vdim: Disabled, this type does not have value dimensions
            **kwargs: Keyword arguments passed to the cloned element

        Returns:
            Cloned object containing the new dimension
        """
        dimension = asdim(dimension)
        if dimension in self.dimensions():
            raise Exception(f'{dimension.name} dimension already defined')
        if vdim and self._deep_indexable:
            raise Exception('Cannot add value dimension to object that is deep indexable')
        if vdim:
            dims = self.vdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(vdims=dims)
            dim_pos += self.ndims
        else:
            dims = self.kdims[:]
            dims.insert(dim_pos, dimension)
            dimensions = dict(kdims=dims)
        if isinstance(dim_val, str) or not hasattr(dim_val, '__iter__'):
            dim_val = cycle([dim_val])
        elif not len(dim_val) == len(self):
            raise ValueError('Added dimension values must be same lengthas existing keys.')
        items = {}
        for dval, (key, val) in zip(dim_val, self.data.items()):
            if vdim:
                new_val = list(val)
                new_val.insert(dim_pos, dval)
                items[key] = tuple(new_val)
            else:
                new_key = list(key)
                new_key.insert(dim_pos, dval)
                items[tuple(new_key)] = val
        return self.clone(items, **dict(dimensions, **kwargs))

    def drop_dimension(self, dimensions):
        """Drops dimension(s) from keys

        Args:
            dimensions: Dimension(s) to drop

        Returns:
            Clone of object with with dropped dimension(s)
        """
        dimensions = [dimensions] if np.isscalar(dimensions) else dimensions
        dims = [d for d in self.kdims if d not in dimensions]
        dim_inds = [self.get_dimension_index(d) for d in dims]
        key_getter = itemgetter(*dim_inds)
        return self.clone([(key_getter(k), v) for k, v in self.data.items()], kdims=dims)

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Args:
            dimension: The dimension to return values for
            expanded (bool, optional): Whether to expand values
                Whether to return the expanded values, behavior depends
                on the type of data:
                  * Columnar: If false returns unique values
                  * Geometry: If false returns scalar values per geometry
                  * Gridded: If false returns 1D coordinates
            flat (bool, optional): Whether to flatten array

        Returns:
            NumPy array of values along the requested dimension
        """
        dimension = self.get_dimension(dimension, strict=True)
        if dimension in self.kdims:
            return np.array([k[self.get_dimension_index(dimension)] for k in self.data.keys()])
        if dimension in self.dimensions():
            values = [el.dimension_values(dimension, expanded, flat) for el in self if dimension in el.dimensions()]
            vals = np.concatenate(values)
            return vals if expanded else util.unique_array(vals)
        else:
            return super().dimension_values(dimension, expanded, flat)

    def reindex(self, kdims=None, force=False):
        """Reindexes object dropping static or supplied kdims

        Creates a new object with a reordered or reduced set of key
        dimensions. By default drops all non-varying key dimensions.

        Reducing the number of key dimensions will discard information
        from the keys. All data values are accessible in the newly
        created object as the new labels must be sufficient to address
        each value uniquely.

        Args:
            kdims (optional): New list of key dimensions after reindexing
            force (bool, optional): Whether to drop non-unique items

        Returns:
            Reindexed object
        """
        if kdims is None:
            kdims = []
        old_kdims = [d.name for d in self.kdims]
        if not isinstance(kdims, list):
            kdims = [kdims]
        elif not len(kdims):
            kdims = [d for d in old_kdims if not len(set(self.dimension_values(d))) == 1]
        indices = [self.get_dimension_index(el) for el in kdims]
        keys = [tuple((k[i] for i in indices)) for k in self.data.keys()]
        reindexed_items = dict(((k, v) for k, v in zip(keys, self.data.values())))
        reduced_dims = {d.name for d in self.kdims}.difference(kdims)
        dimensions = [self.get_dimension(d) for d in kdims if d not in reduced_dims]
        if len(set(keys)) != len(keys) and (not force):
            raise Exception('Given dimension labels not sufficientto address all values uniquely')
        if len(keys):
            cdims = {self.get_dimension(d): self.dimension_values(d)[0] for d in reduced_dims}
        else:
            cdims = {}
        with item_check(indices == sorted(indices)):
            return self.clone(reindexed_items, kdims=dimensions, cdims=cdims)

    @property
    def last(self):
        """Returns the item highest data item along the map dimensions."""
        return list(self.data.values())[-1] if len(self) else None

    @property
    def last_key(self):
        """Returns the last key value."""
        return list(self.keys())[-1] if len(self) else None

    @property
    def info(self):
        """
        Prints information about the Dimensioned object, including the
        number and type of objects contained within it and information
        about its dimensions.
        """
        if len(self.values()) > 0:
            info_str = self.__class__.__name__ + ' containing %d items of type %s\n' % (len(self.keys()), type(self.values()[0]).__name__)
        else:
            info_str = self.__class__.__name__ + ' containing no items\n'
        info_str += '-' * (len(info_str) - 1) + '\n\n'
        aliases = {v: k for k, v in self._dim_aliases.items()}
        for group in self._dim_groups:
            dimensions = getattr(self, group)
            if dimensions:
                group = aliases[group].split('_')[0]
                info_str += f'{group.capitalize()} Dimensions: \n'
            for d in dimensions:
                dmin, dmax = self.range(d.name)
                if d.value_format:
                    dmin, dmax = (d.value_format(dmin), d.value_format(dmax))
                info_str += f'\t {d.pprint_label}: {dmin}...{dmax} \n'
        return info_str

    def update(self, other):
        """Merges other item with this object

        Args:
            other: Object containing items to merge into this object
                Must be a dictionary or NdMapping type
        """
        if isinstance(other, NdMapping):
            dims = [d for d in other.kdims if d not in self.kdims]
            if len(dims) == other.ndims:
                raise KeyError('Cannot update with NdMapping that has a different set of key dimensions.')
            elif dims:
                other = other.drop_dimension(dims)
            other = other.data
        for key, data in other.items():
            self._add_item(key, data, sort=False)
        if self.sort:
            self._resort()

    def keys(self):
        """ Returns the keys of all the elements."""
        if self.ndims == 1:
            return [k[0] for k in self.data.keys()]
        else:
            return list(self.data.keys())

    def values(self):
        """Returns the values of all the elements."""
        return list(self.data.values())

    def items(self):
        """Returns all elements as a list in (key,value) format."""
        return list(zip(list(self.keys()), list(self.values())))

    def get(self, key, default=None):
        """Standard get semantics for all mapping types"""
        try:
            if key is None:
                return None
            return self[key]
        except KeyError:
            return default

    def pop(self, key, default=None):
        """Standard pop semantics for all mapping types"""
        if not isinstance(key, tuple):
            key = (key,)
        return self.data.pop(key, default)

    def __getitem__(self, key):
        """
        Allows multi-dimensional indexing in the order of the
        specified key dimensions, passing any additional indices to
        the data elements.
        """
        if key in [Ellipsis, ()]:
            return self
        map_slice, data_slice = self._split_index(key)
        return self._dataslice(self.data[map_slice], data_slice)

    def __setitem__(self, key, value):
        """Adds item to mapping"""
        self._add_item(key, value, update=False)

    def __str__(self):
        return repr(self)

    def __iter__(self):
        """Iterates over mapping values"""
        return iter(self.values())

    def __contains__(self, key):
        if self.ndims == 1:
            return key in self.data.keys()
        else:
            return key in self.keys()

    def __len__(self):
        return len(self.data)