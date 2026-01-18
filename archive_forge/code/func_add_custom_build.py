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
def add_custom_build(self, node: ET.Element, rulename: str, command: str, deps: T.Optional[T.List[str]]=None, outputs: T.Optional[T.List[str]]=None, msg: T.Optional[str]=None, verify_files: bool=True) -> None:
    igroup = ET.SubElement(node, 'ItemGroup')
    rulefile = os.path.join(self.environment.get_scratch_dir(), rulename + '.rule')
    if not os.path.exists(rulefile):
        with open(rulefile, 'w', encoding='utf-8') as f:
            f.write('# Meson regen file.')
    custombuild = ET.SubElement(igroup, 'CustomBuild', Include=rulefile)
    if msg:
        message = ET.SubElement(custombuild, 'Message')
        message.text = msg
    if not verify_files:
        ET.SubElement(custombuild, 'VerifyInputsAndOutputsExist').text = 'false'
    ET.SubElement(custombuild, 'Command').text = f'{command}\n'
    if not outputs:
        outputs = [self.nonexistent_file(os.path.join(self.environment.get_scratch_dir(), 'outofdate.file'))]
    ET.SubElement(custombuild, 'Outputs').text = ';'.join(outputs)
    if deps:
        ET.SubElement(custombuild, 'AdditionalInputs').text = ';'.join(deps)