import sys
import logging
import re
import tempfile
import xml.etree.ElementTree as ET
import zipfile
import PySide6
from pathlib import Path
from typing import List
from pkginfo import Wheel
from .. import MAJOR_VERSION, BaseConfig, Config, run_command
from . import (create_recipe, find_lib_dependencies, find_qtlibs_in_wheel,
def __find_permissions(self, dependency_files: List[zipfile.Path]):
    permissions = set()
    for dependency_file in dependency_files:
        xml_content = dependency_file.read_text()
        root = ET.fromstring(xml_content)
        for permission in root.iter('permission'):
            permissions.add(permission.attrib['name'])
    return permissions