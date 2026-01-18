import builtins
import datetime as dt
import re
import weakref
from collections import Counter, defaultdict
from collections.abc import Iterable
from functools import partial
from itertools import chain
from operator import itemgetter
import numpy as np
import param
from . import util
from .accessors import Apply, Opts, Redim
from .options import Options, Store, cleanup_custom_options
from .pprint import PrettyPrinter
from .tree import AttrTree
from .util import bytes_to_unicode
class ViewableElement(Dimensioned):
    """
    A ViewableElement is a dimensioned datastructure that may be
    associated with a corresponding atomic visualization. An atomic
    visualization will display the data on a single set of axes
    (i.e. excludes multiple subplots that are displayed at once). The
    only new parameter introduced by ViewableElement is the title
    associated with the object for display.
    """
    __abstract = True
    _auxiliary_component = False
    group = param.String(default='ViewableElement', constant=True)