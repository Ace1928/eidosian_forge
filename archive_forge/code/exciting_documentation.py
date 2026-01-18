import os
import numpy as np
import xml.etree.ElementTree as ET
from ase.io.exciting import atoms2etree
from ase.units import Bohr, Hartree
from ase.calculators.calculator import PropertyNotImplementedError
from xml.dom import minidom

        reads Total energy and forces from info.xml
        