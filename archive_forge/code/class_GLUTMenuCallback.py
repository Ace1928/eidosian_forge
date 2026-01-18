from OpenGL.platform import CurrentContextIsValid, GLUT_GUARD_CALLBACKS, PLATFORM
from OpenGL import contextdata, error, platform, logs
from OpenGL.raw import GLUT as _simple
from OpenGL._bytes import bytes, unicode,as_8_bit
import ctypes, os, sys, traceback
from OpenGL._bytes import long, integer_types
class GLUTMenuCallback(object):
    """Place to collect the GLUT Menu manipulation special code"""
    callbackType = FUNCTION_TYPE(ctypes.c_int, ctypes.c_int)

    def glutCreateMenu(cls, func):
        """Create a new (current) menu, return small integer identifier
        
        func( int ) -- Function taking a single integer reflecting
            the user's choice, the value passed to glutAddMenuEntry
        
        return menuID (small integer)
        """
        cCallback = cls.callbackType(func)
        menu = _simple.glutCreateMenu(cCallback)
        contextdata.setValue(('menucallback', menu), (cCallback, func))
        return menu
    glutCreateMenu.argNames = ['func']
    glutCreateMenu = classmethod(glutCreateMenu)

    def glutDestroyMenu(cls, menu):
        """Destroy (cleanup) the given menu
        
        Deregister's the interal pointer to the menu callback 
        
        returns None
        """
        result = _simple.glutDestroyMenu(menu)
        contextdata.delValue(('menucallback', menu))
        return result
    glutDestroyMenu.argNames = ['menu']
    glutDestroyMenu = classmethod(glutDestroyMenu)