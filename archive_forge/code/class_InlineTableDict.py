import datetime
import io
from os import linesep
import re
import sys
from toml.tz import TomlTz
class InlineTableDict(object):
    """Sentinel subclass of dict for inline tables."""