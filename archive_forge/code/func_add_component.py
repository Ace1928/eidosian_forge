import time
from itertools import chain
from xml.dom import minidom
import xml.etree.ElementTree as ET
def add_component(self, element):
    """Add an element to the entry.

        If the Entry is already part of a pathway, make sure
        the component already exists.
        """
    if self._pathway is not None:
        if element.id not in self._pathway.entries:
            raise ValueError(f'Component {element.id} is not an entry in the pathway')
    self.components.add(element)