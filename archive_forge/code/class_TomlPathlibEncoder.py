import datetime
import re
import sys
from decimal import Decimal
from toml.decoder import InlineTableDict
class TomlPathlibEncoder(TomlEncoder):

    def _dump_pathlib_path(self, v):
        return _dump_str(str(v))

    def dump_value(self, v):
        if (3, 4) <= sys.version_info:
            import pathlib
            if isinstance(v, pathlib.PurePath):
                v = str(v)
        return super(TomlPathlibEncoder, self).dump_value(v)