from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import collections
import six
from pasta.base import annotate
from pasta.base import formatting as fmt
from pasta.base import fstring_utils
class PrintError(Exception):
    """An exception for when we failed to print the tree."""