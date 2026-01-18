from collections import defaultdict, deque
import itertools
import pprint
import textwrap
from jsonschema import _utils
from jsonschema.compat import PY3, iteritems
@property
def absolute_schema_path(self):
    parent = self.parent
    if parent is None:
        return self.relative_schema_path
    path = deque(self.relative_schema_path)
    path.extendleft(reversed(parent.absolute_schema_path))
    return path