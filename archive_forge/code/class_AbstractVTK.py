from __future__ import annotations
import base64
import json
import sys
import zipfile
from abc import abstractmethod
from typing import (
from urllib.request import urlopen
import numpy as np
import param
from bokeh.models import LinearColorMapper
from bokeh.util.serialization import make_globally_unique_id
from pyviz_comms import JupyterComm
from ...param import ParamMethod
from ...util import isfile, lazy_load
from ..base import PaneBase
from ..plot import Bokeh
from .enums import PRESET_CMAPS
class AbstractVTK(PaneBase):
    axes = param.Dict(default={}, nested_refs=True, doc="\n        Parameters of the axes to construct in the 3d view.\n\n        Must contain at least ``xticker``, ``yticker`` and ``zticker``.\n\n        A ``ticker`` is a dictionary which contains:\n          - ``ticks`` (array of numbers) - required.\n              Positions in the scene coordinates of the corresponding\n              axis' ticks.\n          - ``labels`` (array of strings) - optional.\n              Label displayed respectively to the `ticks` positions.\n              If `labels` are not defined they are inferred from the\n              `ticks` array.\n          - ``digits``: number of decimal digits when `ticks` are converted to `labels`.\n          - ``fontsize``: size in pts of the ticks labels.\n          - ``show_grid``: boolean.\n                If true (default) the axes grid is visible.\n          - ``grid_opacity``: float between 0-1.\n                Defines the grid opacity.\n          - ``axes_opacity``: float between 0-1.\n                Defines the axes lines opacity.\n    ")
    camera = param.Dict(nested_refs=True, doc='\n      State of the rendered VTK camera.')
    color_mappers = param.List(nested_refs=True, doc='\n      Color mapper of the actor in the scene')
    orientation_widget = param.Boolean(default=False, doc='\n      Activate/Deactivate the orientation widget display.')
    interactive_orientation_widget = param.Boolean(default=True, constant=True)
    __abstract = True

    def _process_param_change(self, msg):
        msg = super()._process_param_change(msg)
        if 'axes' in msg and msg['axes'] is not None:
            VTKAxes = sys.modules['panel.models.vtk'].VTKAxes
            axes = msg['axes']
            msg['axes'] = VTKAxes(**axes)
        return msg

    def _update_model(self, events: Dict[str, param.parameterized.Event], msg: Dict[str, Any], root: Model, model: Model, doc: Document, comm: Optional[Comm]) -> None:
        if 'axes' in msg and msg['axes'] is not None:
            VTKAxes = sys.modules['panel.models.vtk'].VTKAxes
            axes = msg['axes']
            if isinstance(axes, dict):
                msg['axes'] = VTKAxes(**axes)
            elif isinstance(axes, VTKAxes):
                msg['axes'] = VTKAxes(**axes.properties_with_values())
        super()._update_model(events, msg, root, model, doc, comm)