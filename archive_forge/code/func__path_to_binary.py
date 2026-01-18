import functools
import hashlib
import importlib
import importlib.util
import os
import re
import subprocess
import traceback
from typing import Dict
from ..runtime.driver import DriverBase
def _path_to_binary(binary: str):
    base_dir = os.path.join(os.path.dirname(__file__), os.pardir)
    paths = [os.environ.get(f'TRITON_{binary.upper()}_PATH', ''), os.path.join(base_dir, 'third_party', 'cuda', 'bin', binary)]
    for p in paths:
        bin = p.split(' ')[0]
        if os.path.exists(bin) and os.path.isfile(bin):
            result = subprocess.check_output([bin, '--version'], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search('.*release (\\d+\\.\\d+).*', result.decode('utf-8'), flags=re.MULTILINE)
                if version is not None:
                    return (p, version.group(1))
    raise RuntimeError(f'Cannot find {binary}')