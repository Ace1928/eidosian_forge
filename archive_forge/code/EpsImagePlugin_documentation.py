from __future__ import annotations
import io
import os
import re
import subprocess
import sys
import tempfile
from . import Image, ImageFile
from ._binary import i32le as i32
from ._deprecate import deprecate
EPS Writer for the Python Imaging Library.