from ctypes import *
import pyglet.lib
class struct_xkb_rule_names(Structure):
    _fields_ = [('rules', c_char_p), ('model', c_char_p), ('layout', c_char_p), ('variant', c_char_p), ('options', c_char_p)]