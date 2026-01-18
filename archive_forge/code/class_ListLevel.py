import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
class ListLevel(object):

    def __init__(self, level, sibling_level=True, nested_level=True):
        self.level = level
        self.sibling_level = sibling_level
        self.nested_level = nested_level

    def set_sibling(self, sibling_level):
        self.sibling_level = sibling_level

    def get_sibling(self):
        return self.sibling_level

    def set_nested(self, nested_level):
        self.nested_level = nested_level

    def get_nested(self):
        return self.nested_level

    def set_level(self, level):
        self.level = level

    def get_level(self):
        return self.level