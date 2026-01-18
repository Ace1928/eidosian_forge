import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def _getentry2(self):
    if self._pathway is not None:
        return self._pathway.entries[self._entry2]
    return self._entry2