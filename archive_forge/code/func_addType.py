import copy
import math
import copyreg
import random
import re
import sys
import types
import warnings
from collections import defaultdict, deque
from functools import partial, wraps
from operator import eq, lt
from . import tools  # Needed by HARM-GP
def addType(dict_, ret_type):
    if ret_type not in dict_:
        new_list = []
        for type_, list_ in dict_.items():
            if issubclass(type_, ret_type):
                for item in list_:
                    if item not in new_list:
                        new_list.append(item)
        dict_[ret_type] = new_list