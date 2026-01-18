import os
import sys
from types import ModuleType
from .version import version as __version__  # NOQA:F401
def AliasModule(modname, modpath, attrname=None):
    mod = []

    def getmod():
        if not mod:
            x = importobj(modpath, None)
            if attrname is not None:
                x = getattr(x, attrname)
            mod.append(x)
        return mod[0]
    x = modpath + ('.' + attrname if attrname else '')
    repr_result = '<AliasModule {!r} for {!r}>'.format(modname, x)

    class AliasModule(ModuleType):

        def __repr__(self):
            return repr_result

        def __getattribute__(self, name):
            try:
                return getattr(getmod(), name)
            except ImportError:
                if modpath == 'pytest' and attrname is None:
                    return None
                else:
                    raise

        def __setattr__(self, name, value):
            setattr(getmod(), name, value)

        def __delattr__(self, name):
            delattr(getmod(), name)
    return AliasModule(str(modname))