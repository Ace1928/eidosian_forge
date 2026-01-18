import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import warnings
from typing import (
@cached_property
def _uname_info(self) -> Dict[str, str]:
    if not self.include_uname:
        return {}
    try:
        cmd = ('uname', '-rs')
        stdout = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except OSError:
        return {}
    content = self._to_str(stdout).splitlines()
    return self._parse_uname_content(content)