import collections
import pyrfc3339
from ._conditions import (
def _first_party(name, arg):
    condition = name
    if arg != '':
        condition += ' ' + arg
    return Caveat(condition=condition, namespace=STD_NAMESPACE)