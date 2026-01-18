import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
def SetChild(parent, childname, props):
    Child = ET.SubElement(parent, childname)
    for key in props:
        Child.set(key, props[key])
    return Child