import enum
import functools
import re
import time
from types import SimpleNamespace
import uuid
from weakref import WeakKeyDictionary
import numpy as np
import matplotlib as mpl
from matplotlib._pylab_helpers import Gcf
from matplotlib import _api, cbook
def _get_help_html(self):
    fmt = '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'
    rows = [fmt.format('<b>Action</b>', '<b>Shortcuts</b>', '<b>Description</b>')]
    rows += [fmt.format(*row) for row in self._get_help_entries()]
    return '<style>td {padding: 0px 4px}</style><table><thead>' + rows[0] + '</thead><tbody>'.join(rows[1:]) + '</tbody></table>'