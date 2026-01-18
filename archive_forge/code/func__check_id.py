import re
import sys
import ovs.db.parser
import ovs.db.types
from ovs.db import error
def _check_id(name, json):
    if name.startswith('_'):
        raise error.Error('names beginning with "_" are reserved', json)
    elif not ovs.db.parser.is_identifier(name):
        raise error.Error('name must be a valid id', json)