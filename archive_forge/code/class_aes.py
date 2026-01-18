from __future__ import annotations
import re
import typing
from collections.abc import Iterable, Sequence
from contextlib import suppress
from copy import deepcopy
from dataclasses import fields
from typing import Any, Dict
import pandas as pd
from ..iapi import labels_view
from .evaluation import after_stat, stage
class aes(Dict[str, Any]):
    """
    Create aesthetic mappings

    Parameters
    ----------
    x : str | array_like | scalar
        x aesthetic mapping
    y : str | array_like | scalar
        y aesthetic mapping
    **kwargs : dict
        Other aesthetic mappings

    Notes
    -----
    Only the **x** and **y** aesthetic mappings can be specified as
    positional arguments. All the rest must be keyword arguments.

    The value of each mapping must be one of:

    - **str**

      ```python
       import pandas as pd
       import numpy as np

       arr = [11, 12, 13]
       df = pd.DataFrame({
           "alpha": [1, 2, 3],
           "beta": [1, 2, 3],
           "gam ma": [1, 2, 3]
       })

       # Refer to a column in a dataframe
       ggplot(df, aes(x="alpha", y="beta"))
       ```

    - **array_like**

      ```python
      # A variable
      ggplot(df, aes(x="alpha", y=arr))

      # or an inplace list
      ggplot(df, aes(x="alpha", y=[4, 5, 6]))
      ```

    - **scalar**

      ```python
      # A scalar value/variable
      ggplot(df, aes(x="alpha", y=4))

      # The above statement is equivalent to
      ggplot(df, aes(x="alpha", y=[4, 4, 4]))
      ```

    - **String expression**

      ```python
      ggplot(df, aes(x="alpha", y="2*beta"))
      ggplot(df, aes(x="alpha", y="np.sin(beta)"))
      ggplot(df, aes(x="df.index", y="beta"))

      # If `count` is an aesthetic calculated by a stat
      ggplot(df, aes(x="alpha", y=after_stat("count")))
      ggplot(df, aes(x="alpha", y=after_stat("count/np.max(count)")))
      ```

      The strings in the expression can refer to;

        1. columns in the dataframe
        2. variables in the namespace
        3. aesthetic values (columns) calculated by the `stat`

      with the column names having precedence over the variables.
      For expressions, columns in the dataframe that are mapped to
      must have names that would be valid python variable names.

      This is okay:

      ```python
      # "gam ma" is a column in the dataframe
      ggplot(df, aes(x="df.index", y="gam ma"))
      ```

      While this is not:

      ```python
      # "gam ma" is a column in the dataframe, but not
      # valid python variable name
      ggplot(df, aes(x="df.index", y="np.sin(gam ma)"))
      ```

    `aes` has 2 internal methods you can use to transform variables being
    mapped.

      1. `factor` - This function turns the variable into a factor.
         It is just an alias to `pandas.Categorical`:

         ```python
         ggplot(mtcars, aes(x="factor(cyl)")) + geom_bar()
         ```

      2. `reorder` - This function changes the order of first variable
         based on values of the second variable:

         ```python
         df = pd.DataFrame({
             "x": ["b", "d", "c", "a"],
             "y": [1, 2, 3, 4]
         })

         ggplot(df, aes("reorder(x, y)", "y")) + geom_col()
         ```

    **The group aesthetic**

    `group` is a special aesthetic that the user can *map* to.
    It is used to group the plotted items. If not specified, it
    is automatically computed and in most cases the computed
    groups are sufficient. However, there may be cases were it is
    handy to map to it.

    See Also
    --------
    plotnine.after_stat : For how to map aesthetics to variable
        calculated by the stat
    plotnine.after_scale : For how to alter aesthetics after the
        data has been mapped by the scale.
    plotnine.stage : For how to map to evaluate the mapping to
        aesthetics at more than one stage of the plot building pipeline.
    """

    def __init__(self, *args, **kwargs):
        kwargs = rename_aesthetics(kwargs)
        kwargs.update(zip(('x', 'y'), args))
        kwargs = self._convert_deprecated_expr(kwargs)
        self.update(kwargs)

    def __iter__(self):
        return iter(self.keys())

    def _convert_deprecated_expr(self, kwargs):
        """
        Handle old-style calculated aesthetic expression mappings

        Just converts them to use `stage` e.g.
        "stat(count)" to after_stat(count)
        "..count.." to after_stat(count)
        """
        for name, value in kwargs.items():
            if not isinstance(value, stage) and is_calculated_aes(value):
                _after_stat = strip_calculated_markers(value)
                kwargs[name] = after_stat(_after_stat)
        return kwargs

    @property
    def _starting(self) -> dict[str, Any]:
        """
        Return the subset of aesthetics mapped from the layer data

        The mapping is a dict of the form ``{name: expr}``, i.e the
        stage class has been peeled off.
        """
        d = {}
        for name, value in self.items():
            if not isinstance(value, stage):
                d[name] = value
            elif isinstance(value, stage) and value.start is not None:
                d[name] = value.start
        return d

    @property
    def _calculated(self) -> dict[str, Any]:
        """
        Return only the aesthetics mapped to calculated statistics

        The mapping is a dict of the form ``{name: expr}``, i.e the
        stage class has been peeled off.
        """
        d = {}
        for name, value in self.items():
            if isinstance(value, stage) and value.after_stat is not None:
                d[name] = value.after_stat
        return d

    @property
    def _scaled(self) -> dict[str, Any]:
        """
        Return only the aesthetics mapped to after scaling

        The mapping is a dict of the form ``{name: expr}``, i.e the
        stage class has been peeled off.
        """
        d = {}
        for name, value in self.items():
            if isinstance(value, stage) and value.after_scale is not None:
                d[name] = value.after_scale
        return d

    def __deepcopy__(self, memo):
        """
        Deep copy without copying the environment
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for key, item in self.items():
            result[key] = deepcopy(item, memo)
        return result

    def __radd__(self, plot):
        """
        Add aesthetic mappings to ggplot
        """
        self = deepcopy(self)
        plot.mapping.update(self)
        plot.labels.update(make_labels(self))
        return plot

    def copy(self):
        return aes(**self)

    def inherit(self, other: dict[str, Any] | aes) -> aes:
        """
        Create a  mapping that inherits aesthetics in other

        Parameters
        ----------
        other: aes | dict[str, Any]
            Default aesthetics

        Returns
        -------
        new : aes
            Aesthetic mapping
        """
        new = self.copy()
        for k in other:
            if k not in self:
                new[k] = other[k]
        return new