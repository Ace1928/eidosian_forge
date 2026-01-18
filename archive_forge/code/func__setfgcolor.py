import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def _setfgcolor(self, value):
    if value == 'none':
        self._fgcolor = '#000000'
    else:
        self._fgcolor = value