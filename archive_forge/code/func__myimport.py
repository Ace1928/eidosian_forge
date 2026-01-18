import comhack
import sys
import __builtin__
def _myimport(name, globals=None, locals=None, fromlist=None):
    """
    Tell all modules to imported by McMillan's (or Python's) import.method,
    besides win32com modules automatically genrated by win32com.gencache
    """
    try:
        return mcimport(name, globals, locals, fromlist)
    except ImportError as err:
        if name.startswith('win32com.gen_py'):
            return __oldimport__(name, globals, locals, fromlist)
        else:
            raise err