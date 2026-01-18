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
class BadXml(Exception):
    """
    The XML dictionary registry is not valid XML.
    """