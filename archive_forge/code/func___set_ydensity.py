import numpy as np
from .boundingregion import BoundingBox
from .util import datetime_types
def __set_ydensity(self, density):
    self.__ydensity = density
    self.__ystep = 1.0 / density