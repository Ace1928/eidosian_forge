from datetime import datetime
from boto.compat import six
def _repr_by_type(self, value):
    result = ''
    if isinstance(value, Response):
        result += value.__repr__()
    elif isinstance(value, list):
        result += self._repr_list(value)
    else:
        result += str(value)
    return result