from OpenGL.GL import *  # noqa
import numpy as np
from ...Qt import QtGui
from .. import shaders
from ..GLGraphicsItem import GLGraphicsItem
from ..MeshData import MeshData

        This method must be called to inform the item that the MeshData object
        has been altered.
        