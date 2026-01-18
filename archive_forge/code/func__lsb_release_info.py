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
def _lsb_release_info(self) -> Dict[str, str]:
    """
        Get the information items from the lsb_release command output.

        Returns:
            A dictionary containing all information items.
        """
    if not self.include_lsb:
        return {}
    try:
        cmd = ('lsb_release', '-a')
        stdout = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {}
    content = self._to_str(stdout).splitlines()
    return self._parse_lsb_release_content(content)