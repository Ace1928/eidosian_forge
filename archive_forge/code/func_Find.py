import inspect
import os.path
import sys
from _pydev_bundle._pydev_tipper_common import do_find
from _pydevd_bundle.pydevd_utils import hasattr_checked, dir_checked
from inspect import getfullargspec
def Find(name, log=None):
    f = None
    mod = _imp(name, log)
    parent = mod
    foundAs = ''
    if inspect.ismodule(mod):
        f = get_file(mod)
    components = name.split('.')
    old_comp = None
    for comp in components[1:]:
        try:
            mod = getattr(mod, comp)
        except AttributeError:
            if old_comp != comp:
                raise
        if inspect.ismodule(mod):
            f = get_file(mod)
        else:
            if len(foundAs) > 0:
                foundAs = foundAs + '.'
            foundAs = foundAs + comp
        old_comp = comp
    return (f, mod, parent, foundAs)