import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2FX_REVERB_PARAMETERS(Structure):
    _fields_ = [('WetDryMix', c_float), ('ReflectionsDelay', UINT32), ('ReverbDelay', BYTE), ('RearDelay', UINT32), ('SideDelay', UINT32), ('PositionLeft', BYTE), ('PositionRight', BYTE), ('PositionMatrixLeft', BYTE), ('PositionMatrixRight', BYTE), ('EarlyDiffusion', BYTE), ('LateDiffusion', BYTE), ('LowEQGain', BYTE), ('LowEQCutoff', BYTE), ('LowEQCutoff', BYTE), ('HighEQCutoff', BYTE), ('RoomFilterFreq', c_float), ('RoomFilterMain', c_float), ('RoomFilterHF', c_float), ('ReflectionsGain', c_float), ('ReverbGain', c_float), ('DecayTime', c_float), ('Density', c_float), ('RoomSize', c_float), ('DisableLateField', c_bool)]