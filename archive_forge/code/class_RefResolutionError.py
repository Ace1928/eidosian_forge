from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
class RefResolutionError(Exception):
    pass