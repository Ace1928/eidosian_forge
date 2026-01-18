import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def __set_xdensity(self, density):
    self.__xdensity = density
    self.__xstep = 1.0 / density