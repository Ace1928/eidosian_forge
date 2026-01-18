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
class RepeatChart(TopLevelMixin, core.TopLevelRepeatSpec):
    """A chart repeated across rows and columns with small changes"""

    @utils.use_signature(core.TopLevelRepeatSpec)
    def __init__(self, repeat=Undefined, spec=Undefined, align=Undefined, autosize=Undefined, background=Undefined, bounds=Undefined, center=Undefined, columns=Undefined, config=Undefined, data=Undefined, datasets=Undefined, description=Undefined, name=Undefined, padding=Undefined, params=Undefined, resolve=Undefined, spacing=Undefined, title=Undefined, transform=Undefined, usermeta=Undefined, **kwds):
        _check_if_valid_subspec(spec, 'RepeatChart')
        _spec_as_list = [spec]
        params, _spec_as_list = _combine_subchart_params(params, _spec_as_list)
        spec = _spec_as_list[0]
        if isinstance(spec, (Chart, LayerChart)):
            params = _repeat_names(params, repeat, spec)
        super(RepeatChart, self).__init__(repeat=repeat, spec=spec, align=align, autosize=autosize, background=background, bounds=bounds, center=center, columns=columns, config=config, data=data, datasets=datasets, description=description, name=name, padding=padding, params=params, resolve=resolve, spacing=spacing, title=title, transform=transform, usermeta=usermeta, **kwds)

    def transformed_data(self, row_limit: Optional[int]=None, exclude: Optional[Iterable[str]]=None) -> Optional[DataFrameLike]:
        """Evaluate a RepeatChart's transforms

        Evaluate the data transforms associated with a RepeatChart and return the
        transformed data a DataFrame

        Parameters
        ----------
        row_limit : int (optional)
            Maximum number of rows to return for each DataFrame. None (default) for unlimited
        exclude : iterable of str
            Set of the names of charts to exclude

        Raises
        ------
        NotImplementedError
            RepeatChart does not yet support transformed_data
        """
        raise NotImplementedError('transformed_data is not yet implemented for RepeatChart')

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
        copy = self.copy(deep=False)
        copy.spec = copy.spec.interactive(name=name, bind_x=bind_x, bind_y=bind_y)
        return copy

    def add_params(self, *params: Parameter) -> Self:
        """Add one or more parameters to the chart."""
        if not params or self.spec is Undefined:
            return self
        copy = self.copy()
        copy.spec = copy.spec.add_params(*params)
        return copy.copy()

    @utils.deprecation.deprecated(message="'add_selection' is deprecated. Use 'add_params' instead.")
    def add_selection(self, *selections) -> Self:
        """'add_selection' is deprecated. Use 'add_params' instead."""
        return self.add_params(*selections)