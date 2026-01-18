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
def _distro_release_info(self) -> Dict[str, str]:
    """
        Get the information items from the specified distro release file.

        Returns:
            A dictionary containing all information items.
        """
    if self.distro_release_file:
        distro_info = self._parse_distro_release_file(self.distro_release_file)
        basename = os.path.basename(self.distro_release_file)
        match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
    else:
        try:
            basenames = [basename for basename in os.listdir(self.etc_dir) if basename not in _DISTRO_RELEASE_IGNORE_BASENAMES and os.path.isfile(os.path.join(self.etc_dir, basename))]
            basenames.sort()
        except OSError:
            basenames = _DISTRO_RELEASE_BASENAMES
        for basename in basenames:
            match = _DISTRO_RELEASE_BASENAME_PATTERN.match(basename)
            if match is None:
                continue
            filepath = os.path.join(self.etc_dir, basename)
            distro_info = self._parse_distro_release_file(filepath)
            if 'name' not in distro_info:
                continue
            self.distro_release_file = filepath
            break
        else:
            return {}
    if match is not None:
        distro_info['id'] = match.group(1)
    if 'cloudlinux' in distro_info.get('name', '').lower():
        distro_info['id'] = 'cloudlinux'
    return distro_info