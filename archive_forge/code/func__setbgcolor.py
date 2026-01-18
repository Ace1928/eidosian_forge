import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def _setbgcolor(self, value):
    if value == 'none':
        self._bgcolor = '#000000'
    else:
        self._bgcolor = value