import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
def _dump_int(self, v):
    return '{}'.format(int(v))