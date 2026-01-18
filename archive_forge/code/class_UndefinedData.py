import copy
import enum
from io import StringIO
from math import inf
from pyomo.common.collections import Bunch
class UndefinedData(object):

    def __str__(self):
        return '<undefined>'