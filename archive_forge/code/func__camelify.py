import base64
import json
import math
import os
import re
import struct
import typing
import zlib
from typing import Any, Callable, Union
from jinja2 import Environment, PackageLoader
def _camelify(out):
    return ''.join(['_' + x.lower() if i < len(out) - 1 and x.isupper() and out[i + 1].islower() else x.lower() + '_' if i < len(out) - 1 and x.islower() and out[i + 1].isupper() else x.lower() for i, x in enumerate(list(out))]).lstrip('_').replace('__', '_')