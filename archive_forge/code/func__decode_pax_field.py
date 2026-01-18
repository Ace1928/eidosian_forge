from builtins import open as bltn_open
import sys
import os
import io
import shutil
import stat
import time
import struct
import copy
import re
import warnings
def _decode_pax_field(self, value, encoding, fallback_encoding, fallback_errors):
    """Decode a single field from a pax record.
        """
    try:
        return value.decode(encoding, 'strict')
    except UnicodeDecodeError:
        return value.decode(fallback_encoding, fallback_errors)