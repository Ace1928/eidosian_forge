import ctypes
import sys
from ctypes import *
from ctypes.wintypes import *
from . import com
class LOGFONT(Structure):
    _fields_ = [('lfHeight', LONG), ('lfWidth', LONG), ('lfEscapement', LONG), ('lfOrientation', LONG), ('lfWeight', LONG), ('lfItalic', BYTE), ('lfUnderline', BYTE), ('lfStrikeOut', BYTE), ('lfCharSet', BYTE), ('lfOutPrecision', BYTE), ('lfClipPrecision', BYTE), ('lfQuality', BYTE), ('lfPitchAndFamily', BYTE), ('lfFaceName', c_char * LF_FACESIZE)]