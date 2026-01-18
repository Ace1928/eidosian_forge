from __future__ import annotations
import typing
from ..doctools import document
from .geom_path import geom_path
@document
class geom_line(geom_path):
    """
    Connected points

    {usage}

    Parameters
    ----------
    {common_parameters}

    See Also
    --------
    plotnine.geom_path : For documentation of other parameters.
    """

    def setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.sort_values(['PANEL', 'group', 'x'])