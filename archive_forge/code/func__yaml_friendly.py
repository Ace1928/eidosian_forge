from . import base
from cliff import columns
def _yaml_friendly(value):
    if isinstance(value, columns.FormattableColumn):
        return value.machine_readable()
    elif hasattr(value, 'toDict'):
        return value.toDict()
    elif hasattr(value, 'to_dict'):
        return value.to_dict()
    else:
        return value