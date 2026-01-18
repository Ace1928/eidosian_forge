from __future__ import annotations
import abc
import dataclasses
import datetime
import decimal
from xml.dom import minidom
from xml.etree import ElementTree as ET
def _pretty_xml(element: ET.Element) -> str:
    """Return a pretty formatted XML string representing the given element."""
    return minidom.parseString(ET.tostring(element, encoding='unicode')).toprettyxml()