import configparser
import os
import shlex
import subprocess
from os.path import expanduser, expandvars
from pathlib import Path
from typing import List, Optional, Union
from gitlab.const import USER_AGENT
def _get_values_from_helper(self) -> None:
    """Update attributes that may get values from an external helper program"""
    for attr in HELPER_ATTRIBUTES:
        value = getattr(self, attr)
        if not isinstance(value, str):
            continue
        if not value.lower().strip().startswith(HELPER_PREFIX):
            continue
        helper = value[len(HELPER_PREFIX):].strip()
        commmand = [expanduser(expandvars(token)) for token in shlex.split(helper)]
        try:
            value = subprocess.check_output(commmand, stderr=subprocess.PIPE).decode('utf-8').strip()
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode().strip()
            raise GitlabConfigHelperError(f'Failed to read {attr} value from helper for {self.gitlab_id}:\n{stderr}') from e
        setattr(self, attr, value)