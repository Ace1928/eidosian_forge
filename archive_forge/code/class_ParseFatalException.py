import string
from weakref import ref as wkref
import copy
import sys
import warnings
import re
import sre_constants
import collections
class ParseFatalException(ParseBaseException):
    """user-throwable exception thrown when inconsistent parse content
       is found; stops all parsing immediately"""
    pass