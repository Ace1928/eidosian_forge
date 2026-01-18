from collections import defaultdict
import itertools
import re
import warnings
import sys
from bs4.element import (
from . import _htmlparser
def _initialize_xml_detector(self):
    """Call this method before parsing a document."""
    self._first_processing_instruction = None
    self._root_tag = None