import warnings
from collections import defaultdict
from math import log
def __compute_log_distortion(self):
    if self.__distortion_factor == 0.0:
        self.__log_distortion_factor = log(1e-09)
    else:
        self.__log_distortion_factor = log(self.__distortion_factor)