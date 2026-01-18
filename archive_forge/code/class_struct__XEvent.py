import ctypes
from ctypes import *
import pyglet.lib
class struct__XEvent(Union):
    __slots__ = ['type', 'xany', 'xkey', 'xbutton', 'xmotion', 'xcrossing', 'xfocus', 'xexpose', 'xgraphicsexpose', 'xnoexpose', 'xvisibility', 'xcreatewindow', 'xdestroywindow', 'xunmap', 'xmap', 'xmaprequest', 'xreparent', 'xconfigure', 'xgravity', 'xresizerequest', 'xconfigurerequest', 'xcirculate', 'xcirculaterequest', 'xproperty', 'xselectionclear', 'xselectionrequest', 'xselection', 'xcolormap', 'xclient', 'xmapping', 'xerror', 'xkeymap', 'xgeneric', 'xcookie', 'pad']