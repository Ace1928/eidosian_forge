import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from ase import Atoms
from ase.utils import writer
def CPK_or_BnS(element):
    """Determine how atom is visualized"""
    if element in ['C', 'H', 'O', 'S', 'N']:
        visualization_choice = 'Ball and Stick'
    else:
        visualization_choice = 'CPK'
    return visualization_choice