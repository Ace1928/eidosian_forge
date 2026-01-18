import base64
import copy
import json
import os
import re
import shutil
import sys
import tempfile
import warnings
import zipfile
from io import BytesIO
from xml.dom import minidom
from typing_extensions import deprecated
from selenium.common.exceptions import WebDriverException
def _read_existing_userjs(self, userjs):
    """Reads existing preferences and adds them to desired preference
        dictionary."""
    pref_pattern = re.compile('user_pref\\("(.*)",\\s(.*)\\)')
    with open(userjs, encoding='utf-8') as f:
        for usr in f:
            matches = pref_pattern.search(usr)
            try:
                self._desired_preferences[matches.group(1)] = json.loads(matches.group(2))
            except Exception:
                warnings.warn(f'(skipping) failed to json.loads existing preference: {matches.group(1) + matches.group(2)}')