import collections
import prettytable
from automaton import _utils as utils
from automaton import exceptions as excp
def _orderedkeys(data, sort=True):
    if sort:
        return sorted(data)
    else:
        return list(data)