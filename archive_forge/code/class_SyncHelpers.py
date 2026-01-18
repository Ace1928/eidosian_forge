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
class SyncHelpers:
    """
    Class containing helpers functions to update vtkRenderingWindow
    """

    def make_ren_win(self):
        import vtk
        ren = vtk.vtkRenderer()
        ren_win = vtk.vtkRenderWindow()
        ren_win.AddRenderer(ren)
        return ren_win

    def set_background(self, r, g, b):
        self.get_renderer().SetBackground(r, g, b)
        self.synchronize()

    def add_actors(self, actors):
        """
        Add a list of `actors` to the VTK renderer
        if `reset_camera` is True, the current camera and it's clipping
        will be reset.
        """
        for actor in actors:
            self.get_renderer().AddActor(actor)

    def remove_actors(self, actors):
        """
        Add a list of `actors` to the VTK renderer
        if `reset_camera` is True, the current camera and it's clipping
        will be reset.
        """
        for actor in actors:
            self.get_renderer().RemoveActor(actor)

    def remove_all_actors(self):
        self.remove_actors(self.actors)

    @property
    def vtk_camera(self):
        return self.get_renderer().GetActiveCamera()

    @vtk_camera.setter
    def vtk_camera(self, camera):
        self.get_renderer().SetActiveCamera(camera)

    @property
    def actors(self):
        return list(self.get_renderer().GetActors())

    @abstractmethod
    def synchronize(self):
        """
        function to synchronize the renderer with the view
        """

    @abstractmethod
    def reset_camera(self):
        """
        Reset the camera
        """