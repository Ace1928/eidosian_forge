import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__

        Transform to_data into from_data and return relayout-style
        description of the transformation

        Parameters
        ----------
        to_data : dict|list
        from_data : dict|list

        Returns
        -------
        dict
            relayout-style description of the transformation
        