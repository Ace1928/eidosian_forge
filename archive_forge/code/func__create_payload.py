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
@staticmethod
def _create_payload(payload):
    """Return the string payload filled with zero bytes
           up to the next 512 byte border.
        """
    blocks, remainder = divmod(len(payload), BLOCKSIZE)
    if remainder > 0:
        payload += (BLOCKSIZE - remainder) * NUL
    return payload