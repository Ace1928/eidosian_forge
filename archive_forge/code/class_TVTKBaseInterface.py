import os.path as op
import numpy as np
from numpy import linalg as nla
from .. import logging
from ..interfaces.base import (
from ..interfaces.vtkbase import tvtk
from ..interfaces import vtkbase as VTKInfo
class TVTKBaseInterface(BaseInterface):
    """A base class for interfaces using VTK"""
    _redirect_x = True

    def __init__(self, **inputs):
        if VTKInfo.no_tvtk():
            raise ImportError('This interface requires tvtk to run.')
        super(TVTKBaseInterface, self).__init__(**inputs)