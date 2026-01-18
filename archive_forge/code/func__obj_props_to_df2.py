from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def _obj_props_to_df2(self, obj: Model):
    """ Returns a pandas dataframe of the properties of a bokeh model

        Each row contains  an attribute, its type (a bokeh property), and its docstring.

        """
    obj_dict = obj.properties_with_values()
    types = [obj.lookup(x) for x in obj_dict.keys()]
    docs = [getattr(type(obj), x).__doc__ for x in obj_dict.keys()]
    df = {'props': list(obj_dict.keys()), 'values': list(obj_dict.values()), 'types': types, 'doc': docs}
    return df