from __future__ import annotations
import logging # isort:skip
from itertools import permutations
from typing import TYPE_CHECKING
from bokeh.core.properties import UnsetValueError
from bokeh.layouts import column
from bokeh.models import (
def _make_prop_dict(self) -> pd.DataFrame:
    """ Returns a dataframe containing all the properties of all the submodels of the model being
        analyzed. Used as datasource to show attributes.

        """
    import pandas as pd
    df = pd.DataFrame()
    for x in self._graph.nodes(data=True):
        M = self._model.select_one(dict(id=x[0]))
        Z = pd.DataFrame(self._obj_props_to_df2(M))
        Z['id'] = x[0]
        Z['model'] = str(M)
        Z['values'] = Z['values'].map(lambda x: str(x))
        Z['types'] = Z['types'].map(lambda x: str(x))
        df = pd.concat([df, Z])
    return df