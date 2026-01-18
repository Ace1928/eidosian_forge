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
class VTK:
    """
    The VTK pane renders a VTK scene inside a panel, making it possible to
    interact with complex geometries in 3D.

    Reference: https://panel.holoviz.org/reference/panes/VTK.html

    :Example:

    >>> pn.extension('vtk')
    >>> VTK(some_vtk_object, width=500, height=500)

    This is a Class factory and allows to switch between VTKJS,
    VTKRenderWindow, and VTKRenderWindowSynchronized pane as a function of the
    object type and when the serialisation of the vtkRenderWindow occurs.

    Once a pane is returned by this class (inst = VTK(object)), one can
    use pn.help(inst) to see parameters available for the current pane
    """

    def __new__(self, obj, **params):
        if BaseVTKRenderWindow.applies(obj):
            if VTKRenderWindow.applies(obj, **params):
                return VTKRenderWindow(obj, **params)
            else:
                if params.get('interactive_orientation_widget', False):
                    param.main.param.warning('Setting interactive_orientation_widget=True will break synchronization capabilities of the pane')
                return VTKRenderWindowSynchronized(obj, **params)
        elif VTKJS.applies(obj):
            return VTKJS(obj, **params)

    @staticmethod
    def import_scene(filename, synchronizable=True):
        from .synchronizable_deserializer import import_synch_file
        if synchronizable:
            return VTKRenderWindowSynchronized(import_synch_file(filename=filename), serialize_on_instantiation=False)
        else:
            return VTKRenderWindow(import_synch_file(filename=filename), serialize_on_instantiation=True)