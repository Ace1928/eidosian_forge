import sys
import logging
import os
import re
from xml.etree.ElementTree import parse, SubElement
from .objcreator import QObjectCreator
from .properties import Properties
def addAction(self, elem):
    self.actions.append((self.stack.topwidget, elem.attrib['name']))