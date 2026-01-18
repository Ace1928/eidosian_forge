from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
@staticmethod
def _extract_nmake_fields(captured_build_args: list[str]) -> T.Tuple[str, str, str]:
    include_dir_options = ['-I', '/I', '-isystem', '/clang:-isystem', '/imsvc', '/external:I']
    defs = ''
    paths = '$(VC_IncludePath);$(WindowsSDK_IncludePath);'
    additional_opts = ''
    for arg in captured_build_args:
        if arg.startswith(('-D', '/D')):
            defs += arg[2:] + ';'
        else:
            opt_match = next((opt for opt in include_dir_options if arg.startswith(opt)), None)
            if opt_match:
                paths += arg[len(opt_match):] + ';'
            elif arg.startswith(('-', '/')):
                additional_opts += arg + ' '
    return (defs, paths, additional_opts)