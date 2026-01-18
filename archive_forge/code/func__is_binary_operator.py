import bisect
import configparser
import inspect
import io
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from functools import lru_cache
from optparse import OptionParser
def _is_binary_operator(token_type, text):
    return (token_type == tokenize.OP or text in {'and', 'or'}) and text not in _SYMBOLIC_OPS