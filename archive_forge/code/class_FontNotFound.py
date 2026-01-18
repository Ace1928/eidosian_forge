import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
class FontNotFound(Exception):
    """When there are no usable fonts specified"""