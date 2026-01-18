from rpy2.robjects.packages import importr as _importr
from rpy2.robjects.packages import data
import rpy2.robjects.help as rhelp
from rpy2.rinterface import baseenv
from os import linesep
from collections import OrderedDict
import re
class Packages(object):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __setattr__(self, name, value):
        raise AttributeError("Attributes cannot be set. Use 'importr'")