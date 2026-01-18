from __future__ import annotations
import typing
from warnings import warn
from .._utils.registry import alias
from ..doctools import document
from ..exceptions import PlotnineWarning
from .scale_discrete import scale_discrete
@document
class scale_linetype_manual(_scale_manual):
    """
    Custom discrete linetype scale

    Parameters
    ----------
    values : list | dict
        Linetypes that make up the palette.
        Possible values of the list are:

        1. Strings like

        ```python
        'solid'                # solid line
        'dashed'               # dashed line
        'dashdot'              # dash-dotted line
        'dotted'               # dotted line
        'None' or ' ' or ''    # draw nothing
        ```

        2. Tuples of the form (offset, (on, off, on, off, ....))
           e.g. (0, (1, 1)), (1, (2, 2)), (2, (5, 3, 1, 3))

        The values will be matched with the `limits` of the scale
        or the `breaks` if provided.
        If it is a dict then it should map data values to linetypes.
    {superclass_parameters}

    See Also
    --------
    [](`matplotlib.markers`)
    """
    _aesthetics = ['linetype']

    def map(self, x, limits=None):
        result = super().map(x, limits)
        if len(result) and hasattr(result[0], '__hash__'):
            result = [x if isinstance(x, str) else tuple(x) for x in result]
        return result