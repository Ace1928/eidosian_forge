from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
class TestXMLReader_(object):

    def __init__(self):
        from xml.parsers.expat import ParserCreate
        self.parser = ParserCreate()
        self.parser.StartElementHandler = self.startElement_
        self.parser.EndElementHandler = self.endElement_
        self.parser.CharacterDataHandler = self.addCharacterData_
        self.root = None
        self.stack = []

    def startElement_(self, name, attrs):
        element = (name, attrs, [])
        if self.stack:
            self.stack[-1][2].append(element)
        else:
            self.root = element
        self.stack.append(element)

    def endElement_(self, name):
        self.stack.pop()

    def addCharacterData_(self, data):
        self.stack[-1][2].append(data)