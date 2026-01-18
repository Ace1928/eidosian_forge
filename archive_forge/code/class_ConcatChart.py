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
class ConcatChart(TopLevelMixin, core.TopLevelConcatSpec):
    """A chart with horizontally-concatenated facets"""

    @utils.use_signature(core.TopLevelConcatSpec)
    def __init__(self, data=Undefined, concat=(), columns=Undefined, **kwargs):
        for spec in concat:
            _check_if_valid_subspec(spec, 'ConcatChart')
        super(ConcatChart, self).__init__(data=data, concat=list(concat), columns=columns, **kwargs)
        self.data, self.concat = _combine_subchart_data(self.data, self.concat)
        self.params, self.concat = _combine_subchart_params(self.params, self.concat)

    def __ior__(self, other: core.NonNormalizedSpec) -> Self:
        _check_if_valid_subspec(other, 'ConcatChart')
        self.concat.append(other)
        self.data, self.concat = _combine_subchart_data(self.data, self.concat)
        self.params, self.concat = _combine_subchart_params(self.params, self.concat)
        return self

    def __or__(self, other: core.NonNormalizedSpec) -> Self:
        copy = self.copy(deep=['concat'])
        copy |= other
        return copy

    def transformed_data(self, row_limit: Optional[int]=None, exclude: Optional[Iterable[str]]=None) -> List[DataFrameLike]:
        """Evaluate a ConcatChart's transforms

        Evaluate the data transforms associated with a ConcatChart and return the
        transformed data for each subplot as a list of DataFrames

        Parameters
        ----------
        row_limit : int (optional)
            Maximum number of rows to return for each DataFrame. None (default) for unlimited
        exclude : iterable of str
            Set of the names of charts to exclude

        Returns
        -------
        list of DataFrame
            Transformed data for each subplot as a list of DataFrames
        """
        from altair.utils._transformed_data import transformed_data
        return transformed_data(self, row_limit=row_limit, exclude=exclude)

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
        encodings = []
        if bind_x:
            encodings.append('x')
        if bind_y:
            encodings.append('y')
        return self.add_params(selection_interval(bind='scales', encodings=encodings))

    def add_params(self, *params: Parameter) -> Self:
        """Add one or more parameters to the chart."""
        if not params or not self.concat:
            return self
        copy = self.copy()
        copy.concat = [chart.add_params(*params) for chart in copy.concat]
        return copy

    @utils.deprecation.deprecated(message="'add_selection' is deprecated. Use 'add_params' instead.")
    def add_selection(self, *selections) -> Self:
        """'add_selection' is deprecated. Use 'add_params' instead."""
        return self.add_params(*selections)