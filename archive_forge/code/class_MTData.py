import ctypes
import threading
import collections
import os
from kivy.input.provider import MotionEventProvider
from kivy.input.factory import MotionEventFactory
from kivy.input.motionevent import MotionEvent
from kivy.input.shape import ShapeRect
class MTData(ctypes.Structure):
    _fields_ = [('frame', ctypes.c_int), ('timestamp', ctypes.c_double), ('identifier', ctypes.c_int), ('state', ctypes.c_int), ('unknown1', ctypes.c_int), ('unknown2', ctypes.c_int), ('normalized', MTVector), ('size', ctypes.c_float), ('unknown3', ctypes.c_int), ('angle', ctypes.c_float), ('major_axis', ctypes.c_float), ('minor_axis', ctypes.c_float), ('unknown4', MTVector), ('unknown5_1', ctypes.c_int), ('unknown5_2', ctypes.c_int), ('unknown6', ctypes.c_float)]