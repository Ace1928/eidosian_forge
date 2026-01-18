from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_continuous import scale_continuous
from .scale_datetime import scale_datetime
from .scale_discrete import scale_discrete
@document
class scale_color_cmap_d(scale_discrete):
    """
    A discrete color scales using Matplotlib colormaps

    Parameters
    ----------
    cmap_name :
        A standard Matplotlib colormap name. It must be of type
        [](`matplotlib.colors.ListedColormap`).
        The default is `viridis`. For the list of names checkout
        the output of `matplotlib.cm.cmap_d.keys()` or see the
        `documentation <http://matplotlib.org/users/colormaps.html>`_.
    {superclass_parameters}
    na_value : str, default="#7F7F7F"
        Color of missing values.

    See Also
    --------
    [](`matplotlib.cm`)
    [](`matplotlib.colors`)
    """
    _aesthetics = ['color']
    na_value = '#7F7F7F'

    def __init__(self, cmap_name: str='viridis', **kwargs):
        from mizani.palettes import cmap_d_pal
        self._palette = cmap_d_pal(cmap_name)
        super().__init__(**kwargs)