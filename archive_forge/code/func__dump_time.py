import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
def _dump_time(v):
    utcoffset = v.utcoffset()
    if utcoffset is None:
        return v.isoformat()
    return v.isoformat()[:-6]