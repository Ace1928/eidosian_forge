from __future__ import annotations
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING
import numpy as np
import scipy.constants as const
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import Element, Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.bandstructure import HighSymmKpath
def _dicttoxml(self, paramdict_, element):
    for key, value in paramdict_.items():
        if isinstance(value, str) and key == 'text()':
            element.text = value
        elif isinstance(value, str):
            element.attrib[key] = value
        elif isinstance(value, list):
            for item in value:
                self._dicttoxml(item, ET.SubElement(element, key))
        elif isinstance(value, dict):
            if element.findall(key) == []:
                self._dicttoxml(value, ET.SubElement(element, key))
            else:
                self._dicttoxml(value, element.findall(key)[0])
        else:
            print('cannot deal with', key, '=', value)