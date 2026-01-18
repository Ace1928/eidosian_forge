import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class X3DAUDIO_EMITTER(Structure):
    _fields_ = [('pCone', POINTER(X3DAUDIO_CONE)), ('OrientFront', X3DAUDIO_VECTOR), ('OrientTop', X3DAUDIO_VECTOR), ('Position', X3DAUDIO_VECTOR), ('Velocity', X3DAUDIO_VECTOR), ('InnerRadius', FLOAT32), ('InnerRadiusAngle', FLOAT32), ('ChannelCount', UINT32), ('ChannelRadius', FLOAT32), ('pChannelAzimuths', POINTER(FLOAT32)), ('pVolumeCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)), ('pLFECurve', POINTER(X3DAUDIO_DISTANCE_CURVE)), ('pLPFDirectCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)), ('pLPFReverbCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)), ('pReverbCurve', POINTER(X3DAUDIO_DISTANCE_CURVE)), ('CurveDistanceScaler', FLOAT32), ('DopplerScaler', FLOAT32)]