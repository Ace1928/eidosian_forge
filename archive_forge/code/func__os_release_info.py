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
def _os_release_info(self) -> Dict[str, str]:
    """
        Get the information items from the specified os-release file.

        Returns:
            A dictionary containing all information items.
        """
    if os.path.isfile(self.os_release_file):
        with open(self.os_release_file, encoding='utf-8') as release_file:
            return self._parse_os_release_content(release_file)
    return {}