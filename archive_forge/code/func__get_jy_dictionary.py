from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
def _get_jy_dictionary(self, obj):
    ret = {}
    found = java.util.HashMap()
    original = obj
    if hasattr_checked(obj, '__class__') and obj.__class__ == java.lang.Class:
        classes = []
        classes.append(obj)
        c = obj.getSuperclass()
        while c != None:
            classes.append(c)
            c = c.getSuperclass()
        interfs = []
        for obj in classes:
            interfs.extend(obj.getInterfaces())
        classes.extend(interfs)
        for obj in classes:
            declaredMethods = obj.getDeclaredMethods()
            declaredFields = obj.getDeclaredFields()
            for i in range(len(declaredMethods)):
                name = declaredMethods[i].getName()
                ret[name] = declaredMethods[i].toString()
                found.put(name, 1)
            for i in range(len(declaredFields)):
                name = declaredFields[i].getName()
                found.put(name, 1)
                declaredFields[i].setAccessible(True)
                try:
                    ret[name] = declaredFields[i].get(original)
                except:
                    ret[name] = declaredFields[i].toString()
    try:
        d = dir(original)
        for name in d:
            if found.get(name) != 1:
                ret[name] = getattr(original, name)
    except:
        pass
    return ret