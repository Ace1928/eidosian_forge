import numpy as np
import param
from . import traversal
from .dimension import Dimension, Dimensioned, ViewableElement, ViewableTree
from .ndmapping import NdMapping, UniformNdMapping
class AdjointLayout(Layoutable, Dimensioned):
    """
    An AdjointLayout provides a convenient container to lay out some
    marginal plots next to a primary plot. This is often useful to
    display the marginal distributions of a plot next to the primary
    plot. AdjointLayout accepts a list of up to three elements, which
    are laid out as follows with the names 'main', 'top' and 'right':

     _______________
    |     3     |   |
    |___________|___|
    |           |   |  1:  main
    |           |   |  2:  right
    |     1     | 2 |  3:  top
    |           |   |
    |___________|___|
    """
    kdims = param.List(default=[Dimension('AdjointLayout')], constant=True)
    layout_order = ['main', 'right', 'top']
    _deep_indexable = True
    _auxiliary_component = False

    def __init__(self, data, **params):
        self.main_layer = 0
        if data and len(data) > 3:
            raise Exception('AdjointLayout accepts no more than three elements.')
        if data is not None and all((isinstance(v, tuple) for v in data)):
            data = dict(data)
        if isinstance(data, dict):
            wrong_pos = [k for k in data if k not in self.layout_order]
            if wrong_pos:
                raise Exception('Wrong AdjointLayout positions provided.')
        elif isinstance(data, list):
            data = dict(zip(self.layout_order, data))
        else:
            data = {}
        super().__init__(data, **params)

    def __mul__(self, other, reverse=False):
        layer1 = other if reverse else self
        layer2 = self if reverse else other
        adjoined_items = []
        if isinstance(layer1, AdjointLayout) and isinstance(layer2, AdjointLayout):
            adjoined_items = []
            adjoined_items.append(layer1.main * layer2.main)
            if layer1.right is not None and layer2.right is not None:
                if layer1.right.dimensions() == layer2.right.dimensions():
                    adjoined_items.append(layer1.right * layer2.right)
                else:
                    adjoined_items += [layer1.right, layer2.right]
            elif layer1.right is not None:
                adjoined_items.append(layer1.right)
            elif layer2.right is not None:
                adjoined_items.append(layer2.right)
            if layer1.top is not None and layer2.top is not None:
                if layer1.top.dimensions() == layer2.top.dimensions():
                    adjoined_items.append(layer1.top * layer2.top)
                else:
                    adjoined_items += [layer1.top, layer2.top]
            elif layer1.top is not None:
                adjoined_items.append(layer1.top)
            elif layer2.top is not None:
                adjoined_items.append(layer2.top)
            if len(adjoined_items) > 3:
                raise ValueError('AdjointLayouts could not be overlaid, the dimensions of the adjoined plots do not match and the AdjointLayout can hold no more than two adjoined plots.')
        elif isinstance(layer1, AdjointLayout):
            adjoined_items = [layer1.data[o] for o in self.layout_order if o in layer1.data]
            adjoined_items[0] = layer1.main * layer2
        elif isinstance(layer2, AdjointLayout):
            adjoined_items = [layer2.data[o] for o in self.layout_order if o in layer2.data]
            adjoined_items[0] = layer1 * layer2.main
        if adjoined_items:
            return self.clone(adjoined_items)
        else:
            return NotImplemented

    def __rmul__(self, other):
        return self.__mul__(other, reverse=True)

    @property
    def group(self):
        """Group inherited from main element"""
        if self.main and self.main.group != type(self.main).__name__:
            return self.main.group
        else:
            return 'AdjointLayout'

    @property
    def label(self):
        """Label inherited from main element"""
        return self.main.label if self.main else ''

    @group.setter
    def group(self, group):
        pass

    @label.setter
    def label(self, label):
        pass

    def relabel(self, label=None, group=None, depth=1):
        """Clone object and apply new group and/or label.

        Applies relabeling to child up to the supplied depth.

        Args:
            label (str, optional): New label to apply to returned object
            group (str, optional): New group to apply to returned object
            depth (int, optional): Depth to which relabel will be applied
                If applied to container allows applying relabeling to
                contained objects up to the specified depth

        Returns:
            Returns relabelled object
        """
        return super().relabel(label=label, group=group, depth=depth)

    def get(self, key, default=None):
        """
        Returns the viewable corresponding to the supplied string
        or integer based key.

        Args:
            key: Numeric or string index: 0) 'main' 1) 'right' 2) 'top'
            default: Value returned if key not found

        Returns:
            Indexed value or supplied default
        """
        return self.data[key] if key in self.data else default

    def dimension_values(self, dimension, expanded=True, flat=True):
        """Return the values along the requested dimension.

        Applies to the main object in the AdjointLayout.

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
        dimension = self.get_dimension(dimension, strict=True).name
        return self.main.dimension_values(dimension, expanded, flat)

    def __getitem__(self, key):
        """Index into the AdjointLayout by index or label"""
        if key == ():
            return self
        data_slice = None
        if isinstance(key, tuple):
            data_slice = key[1:]
            key = key[0]
        if isinstance(key, int) and key <= len(self):
            if key == 0:
                data = self.main
            if key == 1:
                data = self.right
            if key == 2:
                data = self.top
            if data_slice:
                data = data[data_slice]
            return data
        elif isinstance(key, str) and key in self.data:
            if data_slice is None:
                return self.data[key]
            else:
                self.data[key][data_slice]
        elif isinstance(key, slice) and key.start is None and (key.stop is None):
            return self if data_slice is None else self.clone([el[data_slice] for el in self])
        else:
            raise KeyError(f'Key {key} not found in AdjointLayout.')

    def __setitem__(self, key, value):
        if key in ['main', 'right', 'top']:
            if isinstance(value, (ViewableElement, UniformNdMapping, Empty)):
                self.data[key] = value
            else:
                raise ValueError('AdjointLayout only accepts Element types.')
        else:
            raise Exception(f'Position {key} not valid in AdjointLayout.')

    def __lshift__(self, other):
        """Add another plot to the AdjointLayout"""
        views = [self.data.get(k, None) for k in self.layout_order]
        return AdjointLayout([v for v in views if v is not None] + [other])

    @property
    def ddims(self):
        return self.main.dimensions()

    @property
    def main(self):
        """Returns the main element in the AdjointLayout"""
        return self.data.get('main', None)

    @property
    def right(self):
        """Returns the right marginal element in the AdjointLayout"""
        return self.data.get('right', None)

    @property
    def top(self):
        """Returns the top marginal element in the AdjointLayout"""
        return self.data.get('top', None)

    @property
    def last(self):
        items = [(k, v.last) if isinstance(v, NdMapping) else (k, v) for k, v in self.data.items()]
        return self.__class__(dict(items))

    def keys(self):
        return list(self.data.keys())

    def items(self):
        return list(self.data.items())

    def __iter__(self):
        i = 0
        while i < len(self):
            yield self[i]
            i += 1

    def __len__(self):
        """Number of items in the AdjointLayout"""
        return len(self.data)