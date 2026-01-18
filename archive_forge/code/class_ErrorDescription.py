import textwrap
import os
import pkg_resources
from .command import Command, BadCommand
import fnmatch
import re
import traceback
from io import StringIO
import inspect
import types
class ErrorDescription(object):

    def __init__(self, exc, tb):
        self.exc = exc
        self.tb = '\n'.join(tb)
        self.description = 'Error loading: %s' % exc