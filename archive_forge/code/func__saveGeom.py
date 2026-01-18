import logging
from reportlab import rl_config
def _saveGeom(self, **kwds):
    if not self.__dict__.setdefault('_savedGeom', {}):
        for ga in _geomAttr:
            ga = '_' + ga
            self.__dict__['_savedGeom'][ga] = self.__dict__[ga]
    for k, v in kwds.items():
        setattr(self, k, v)