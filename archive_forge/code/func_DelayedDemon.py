from sys import version_info as _swig_python_version_info
import weakref
def DelayedDemon(self, method, *args):
    demon = PyConstraintDemon(self, method, True, *args)
    self.__demons.append(demon)
    return demon