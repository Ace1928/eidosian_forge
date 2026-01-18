import functools
import gettext
import logging
import os
import shutil
import sys
import warnings
import xml.dom.minidom
import xml.parsers.expat
import zipfile
class BadExtensionFile(Exception):
    """
    The extension has a wrong file format, should be a ZIP file.
    """