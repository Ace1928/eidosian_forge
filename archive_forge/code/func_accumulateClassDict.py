import os
import pickle
import re
import sys
import traceback
import types
import weakref
from collections import deque
from io import IOBase, StringIO
from typing import Type, Union
from twisted.python.compat import nativeString
from twisted.python.deprecate import _fullyQualifiedName as fullyQualifiedName
def accumulateClassDict(classObj, attr, adict, baseClass=None):
    """
    Accumulate all attributes of a given name in a class hierarchy into a single dictionary.

    Assuming all class attributes of this name are dictionaries.
    If any of the dictionaries being accumulated have the same key, the
    one highest in the class hierarchy wins.
    (XXX: If "highest" means "closest to the starting class".)

    Ex::

      class Soy:
        properties = {"taste": "bland"}

      class Plant:
        properties = {"colour": "green"}

      class Seaweed(Plant):
        pass

      class Lunch(Soy, Seaweed):
        properties = {"vegan": 1 }

      dct = {}

      accumulateClassDict(Lunch, "properties", dct)

      print(dct)

    {"taste": "bland", "colour": "green", "vegan": 1}
    """
    for base in classObj.__bases__:
        accumulateClassDict(base, attr, adict)
    if baseClass is None or baseClass in classObj.__bases__:
        adict.update(classObj.__dict__.get(attr, {}))