import collections
import os
import re
import sys
import functools
import itertools
def _java_getprop(name, default):
    from java.lang import System
    try:
        value = System.getProperty(name)
        if value is None:
            return default
        return value
    except AttributeError:
        return default