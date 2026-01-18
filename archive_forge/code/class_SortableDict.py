import sys
import os
import time
import re
import string
import urllib.request, urllib.parse, urllib.error
from docutils import frontend, nodes, languages, writers, utils, io
from docutils.utils.error_reporting import SafeString
from docutils.transforms import writer_aux
from docutils.utils.math import pick_math_environment, unichar2tex
class SortableDict(dict):
    """Dictionary with additional sorting methods

    Tip: use key starting with with '_' for sorting before small letters
         and with '~' for sorting after small letters.
    """

    def sortedkeys(self):
        """Return sorted list of keys"""
        keys = list(self.keys())
        keys.sort()
        return keys

    def sortedvalues(self):
        """Return list of values sorted by keys"""
        return [self[key] for key in self.sortedkeys()]