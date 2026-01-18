from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
class Collator(NdMapping):
    """
    Collator is an NdMapping type which can merge any number
    of HoloViews components with whatever level of nesting
    by inserting the Collators key dimensions on the HoloMaps.
    If the items in the Collator do not contain HoloMaps
    they will be created. Collator also supports filtering
    of Tree structures and dropping of constant dimensions.
    """
    drop = param.List(default=[], doc='\n        List of dimensions to drop when collating data, specified\n        as strings.')
    drop_constant = param.Boolean(default=False, doc='\n        Whether to demote any non-varying key dimensions to\n        constant dimensions.')
    filters = param.List(default=[], doc='\n        List of paths to drop when collating data, specified\n        as strings or tuples.')
    group = param.String(default='Collator')
    progress_bar = param.Parameter(default=None, doc='\n         The progress bar instance used to report progress. Set to\n         None to disable progress bars.')
    merge_type = param.ClassSelector(class_=NdMapping, default=HoloMap, is_instance=False, instantiate=False)
    value_transform = param.Callable(default=None, doc='\n        If supplied the function will be applied on each Collator\n        value during collation. This may be used to apply an operation\n        to the data or load references from disk before they are collated\n        into a displayable HoloViews object.')
    vdims = param.List(default=[], doc='\n         Collator operates on HoloViews objects, if vdims are specified\n         a value_transform function must also be supplied.')
    _deep_indexable = False
    _auxiliary_component = False
    _nest_order = {HoloMap: ViewableElement, GridSpace: (HoloMap, CompositeOverlay, ViewableElement), NdLayout: (GridSpace, HoloMap, ViewableElement), NdOverlay: Element}

    def __init__(self, data=None, **params):
        if isinstance(data, Element):
            params = dict(get_param_values(data), **params)
            if 'kdims' not in params:
                params['kdims'] = data.kdims
            if 'vdims' not in params:
                params['vdims'] = data.vdims
            data = data.mapping()
        super().__init__(data, **params)

    def __call__(self):
        """
        Filter each Layout in the Collator with the supplied
        path_filters. If merge is set to True all Layouts are
        merged, otherwise an NdMapping containing all the
        Layouts is returned. Optionally a list of dimensions
        to be ignored can be supplied.
        """
        constant_dims = self.static_dimensions
        ndmapping = NdMapping(kdims=self.kdims)
        num_elements = len(self)
        for idx, (key, data) in enumerate(self.data.items()):
            if isinstance(data, AttrTree):
                data = data.filter(self.filters)
            if len(self.vdims) and self.value_transform:
                vargs = dict(zip(self.dimensions('value', label=True), data))
                data = self.value_transform(vargs)
            if not isinstance(data, Dimensioned):
                raise ValueError('Collator values must be Dimensioned objects before collation.')
            dim_keys = zip(self.kdims, key)
            varying_keys = [(d, k) for d, k in dim_keys if not self.drop_constant or (d not in constant_dims and d not in self.drop)]
            constant_keys = [(d, k) for d, k in dim_keys if d in constant_dims and d not in self.drop and self.drop_constant]
            if varying_keys or constant_keys:
                data = self._add_dimensions(data, varying_keys, dict(constant_keys))
            ndmapping[key] = data
            if self.progress_bar is not None:
                self.progress_bar(float(idx + 1) / num_elements * 100)
        components = ndmapping.values()
        accumulator = ndmapping.last.clone(components[0].data)
        for component in components:
            accumulator.update(component)
        return accumulator

    @property
    def static_dimensions(self):
        """
        Return all constant dimensions.
        """
        dimensions = []
        for dim in self.kdims:
            if len(set(self.dimension_values(dim.name))) == 1:
                dimensions.append(dim)
        return dimensions

    def _add_dimensions(self, item, dims, constant_keys):
        """
        Recursively descend through an Layout and NdMapping objects
        in order to add the supplied dimension values to all contained
        HoloMaps.
        """
        if isinstance(item, Layout):
            item.fixed = False
        dim_vals = [(dim, val) for dim, val in dims[::-1] if dim not in self.drop]
        if isinstance(item, self.merge_type):
            new_item = item.clone(cdims=constant_keys)
            for dim, val in dim_vals:
                dim = asdim(dim)
                if dim not in new_item.kdims:
                    new_item = new_item.add_dimension(dim, 0, val)
        elif isinstance(item, self._nest_order[self.merge_type]):
            if len(dim_vals):
                dimensions, key = zip(*dim_vals)
                new_item = self.merge_type({key: item}, kdims=list(dimensions), cdims=constant_keys)
            else:
                new_item = item
        else:
            new_item = item.clone(shared_data=False, cdims=constant_keys)
            for k, v in item.items():
                new_item[k] = self._add_dimensions(v, dims[::-1], constant_keys)
        if isinstance(new_item, Layout):
            new_item.fixed = True
        return new_item