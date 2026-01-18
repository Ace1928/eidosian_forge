import sys
import uuid
import ovs.db.data
import ovs.db.parser
import ovs.ovsuuid
from ovs.db import error
@staticmethod
def __n_from_json(json, default):
    if json is None:
        return default
    elif isinstance(json, int) and 0 <= json <= sys.maxsize:
        return json
    else:
        raise error.Error('bad min or max value', json)