from .. import errors
from ..osutils import basename
from ..revision import NULL_REVISION
from . import inventory
def _deserialize_bool(self, value):
    if value == b'true':
        return True
    elif value == b'false':
        return False
    else:
        raise InventoryDeltaError('value %(val)r is not a bool', val=value)