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
def _prettyprint_vcxproj_xml(self, tree: ET.ElementTree, ofname: str) -> None:
    ofname_tmp = ofname + '~'
    tree.write(ofname_tmp, encoding='utf-8', xml_declaration=True)
    doc = xml.dom.minidom.parse(ofname_tmp)
    with open(ofname_tmp, 'w', encoding='utf-8') as of:
        of.write(doc.toprettyxml())
    replace_if_different(ofname, ofname_tmp)