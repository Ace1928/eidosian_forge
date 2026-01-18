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
class VTKJS(AbstractVTK):
    """
    The VTKJS pane allow rendering a vtk scene stored in a vtkjs.

    Reference: https://panel.holoviz.org/reference/panes/VTKJS.html

    :Example:

    >>> pn.extension('vtk')
    >>> VTK(
    ...    'https://raw.githubusercontent.com/Kitware/vtk-js/master/Data/StanfordDragon.vtkjs',
    ...     sizing_mode='stretch_width', height=400, enable_keybindings=True,
    ...     orientation_widget=True
    ... )
    """
    enable_keybindings = param.Boolean(default=False, doc='\n        Activate/Deactivate keys binding.\n\n        Warning: These keybindings may not work as expected in a\n                 notebook context if they interact with already\n                 bound keys.')
    _serializers = {}
    _updates = True

    def __init__(self, object=None, **params):
        super().__init__(object, **params)
        self._vtkjs = None

    @classmethod
    def applies(cls, obj: Any) -> float | bool | None:
        if isinstance(obj, str) and obj.endswith('.vtkjs'):
            return True

    def _get_model(self, doc: Document, root: Optional[Model]=None, parent: Optional[Model]=None, comm: Optional[Comm]=None) -> Model:
        """
        Should return the bokeh model to be rendered.
        """
        VTKJSPlot = lazy_load('panel.models.vtk', 'VTKJSPlot', isinstance(comm, JupyterComm), root)
        props = self._get_properties(doc)
        props['data_url'], props['data'] = self._get_vtkjs()
        model = VTKJSPlot(**props)
        root = root or model
        self._link_props(model, ['camera', 'enable_keybindings', 'orientation_widget'], doc, root, comm)
        self._models[root.ref['id']] = (model, parent)
        return model

    def _get_vtkjs(self, fetch=True):
        data_path, data_url = (None, None)
        if isinstance(self.object, str) and self.object.endswith('.vtkjs'):
            data_path = data_path
            if not isfile(self.object):
                data_url = self.object
        if self._vtkjs is None and self.object is not None:
            vtkjs = None
            if data_url and fetch:
                vtkjs = urlopen(data_url).read() if fetch else data_url
            elif data_path:
                with open(self.object, 'rb') as f:
                    vtkjs = f.read()
            elif hasattr(self.object, 'read'):
                vtkjs = self.object.read()
            self._vtkjs = vtkjs
        return (data_url, self._vtkjs)

    def _update(self, ref: str, model: Model) -> None:
        self._vtkjs = None
        data_url, vtkjs = self._get_vtkjs()
        model.update(data_url=data_url, data=vtkjs)

    def export_vtkjs(self, filename: str | IO='vtk_panel.vtkjs'):
        """
        Exports current VTK data to .vtkjs file.

        Arguments
        ---------
        filename: str | IO
        """
        _, vtkjs = self._get_vtkjs()
        if hasattr(filename, 'write'):
            filename.write(vtkjs)
        else:
            with open(filename, 'wb') as f:
                f.write(vtkjs)