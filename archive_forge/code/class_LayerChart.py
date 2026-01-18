import warnings
import hashlib
import io
import json
import jsonschema
import pandas as pd
from toolz.curried import pipe as _pipe
import itertools
import sys
from typing import cast, List, Optional, Any, Iterable, Union, Literal, IO
from typing import Type as TypingType
from typing import Dict as TypingDict
from .schema import core, channels, mixins, Undefined, UndefinedType, SCHEMA_URL
from .data import data_transformers
from ... import utils, expr
from ...expr import core as _expr_core
from .display import renderers, VEGALITE_VERSION, VEGAEMBED_VERSION, VEGA_VERSION
from .theme import themes
from .compiler import vegalite_compilers
from ...utils._vegafusion_data import (
from ...utils.core import DataFrameLike
from ...utils.data import DataType
class LayerChart(TopLevelMixin, _EncodingMixin, core.TopLevelLayerSpec):
    """A Chart with layers within a single panel"""

    @utils.use_signature(core.TopLevelLayerSpec)
    def __init__(self, data=Undefined, layer=(), **kwargs):
        for spec in layer:
            _check_if_valid_subspec(spec, 'LayerChart')
            _check_if_can_be_layered(spec)
        super(LayerChart, self).__init__(data=data, layer=list(layer), **kwargs)
        self.data, self.layer = _combine_subchart_data(self.data, self.layer)
        self.layer = _remove_duplicate_params(self.layer)
        self.params, self.layer = _combine_subchart_params(self.params, self.layer)
        layer_props = ('height', 'width', 'view')
        combined_dict, self.layer = _remove_layer_props(self, self.layer, layer_props)
        for prop in combined_dict:
            self[prop] = combined_dict[prop]

    def transformed_data(self, row_limit: Optional[int]=None, exclude: Optional[Iterable[str]]=None) -> List[DataFrameLike]:
        """Evaluate a LayerChart's transforms

        Evaluate the data transforms associated with a LayerChart and return the
        transformed data for each layer as a list of DataFrames

        Parameters
        ----------
        row_limit : int (optional)
            Maximum number of rows to return for each DataFrame. None (default) for unlimited
        exclude : iterable of str
            Set of the names of charts to exclude

        Returns
        -------
        list of DataFrame
            Transformed data for each layer as a list of DataFrames
        """
        from altair.utils._transformed_data import transformed_data
        return transformed_data(self, row_limit=row_limit, exclude=exclude)

    def __iadd__(self, other: Union[core.LayerSpec, core.UnitSpec]) -> Self:
        _check_if_valid_subspec(other, 'LayerChart')
        _check_if_can_be_layered(other)
        self.layer.append(other)
        self.data, self.layer = _combine_subchart_data(self.data, self.layer)
        self.params, self.layer = _combine_subchart_params(self.params, self.layer)
        return self

    def __add__(self, other: Union[core.LayerSpec, core.UnitSpec]) -> Self:
        copy = self.copy(deep=['layer'])
        copy += other
        return copy

    def add_layers(self, *layers: Union[core.LayerSpec, core.UnitSpec]) -> Self:
        copy = self.copy(deep=['layer'])
        for layer in layers:
            copy += layer
        return copy

    def interactive(self, name: Optional[str]=None, bind_x: bool=True, bind_y: bool=True) -> Self:
        """Make chart axes scales interactive

        Parameters
        ----------
        name : string
            The parameter name to use for the axes scales. This name should be
            unique among all parameters within the chart.
        bind_x : boolean, default True
            If true, then bind the interactive scales to the x-axis
        bind_y : boolean, default True
            If true, then bind the interactive scales to the y-axis

        Returns
        -------
        chart :
            copy of self, with interactive axes added

        """
        if not self.layer:
            raise ValueError('LayerChart: cannot call interactive() until a layer is defined')
        copy = self.copy(deep=['layer'])
        copy.layer[0] = copy.layer[0].interactive(name=name, bind_x=bind_x, bind_y=bind_y)
        return copy

    def add_params(self, *params: Parameter) -> Self:
        """Add one or more parameters to the chart."""
        if not params or not self.layer:
            return self
        copy = self.copy()
        copy.layer[0] = copy.layer[0].add_params(*params)
        return copy.copy()

    @utils.deprecation.deprecated(message="'add_selection' is deprecated. Use 'add_params' instead.")
    def add_selection(self, *selections) -> Self:
        """'add_selection' is deprecated. Use 'add_params' instead."""
        return self.add_params(*selections)