from __future__ import (absolute_import, division, print_function)
import importlib
import os
import re
import sys
import textwrap
import yaml
class DocFragmentParseError(Exception):

    def __init__(self, path, error_message):
        self.path = path
        self.error_message = error_message
        super(DocFragmentParseError, self).__init__('Error while parsing {0}: {1}'.format(path, error_message))