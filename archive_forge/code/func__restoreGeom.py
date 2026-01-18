import logging
from reportlab import rl_config
def _restoreGeom(self):
    if self.__dict__.get('_savedGeom', None):
        for ga in _geomAttr:
            ga = '_' + ga
            self.__dict__[ga] = self.__dict__[ga]['_savedGeom']
            del self.__dict__['_savedGeom']
        self._geom()