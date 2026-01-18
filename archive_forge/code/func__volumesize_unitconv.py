import argparse
import getpass
import inspect
import io
import lzma
import os
import pathlib
import platform
import re
import shutil
import sys
from lzma import CHECK_CRC64, CHECK_SHA256, is_check_supported
from typing import Any, List, Optional
import _lzma  # type: ignore
import multivolumefile
import texttable  # type: ignore
import py7zr
from py7zr.callbacks import ExtractCallback
from py7zr.compressor import SupportedMethods
from py7zr.helpers import Local
from py7zr.properties import COMMAND_HELP_STRING
def _volumesize_unitconv(self, size: str) -> int:
    m = self.unit_pattern.match(size)
    if m is not None:
        num = m.group(1)
        unit = m.group(2)
        return int(num) if unit is None else int(num) * self.dunits[unit]
    else:
        return -1