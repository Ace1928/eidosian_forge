import zipfile
import six
import logging
import uuid
import warnings
import posixpath as zip_path
import os.path
from collections import OrderedDict
from lxml import etree
import ebooklib
from ebooklib.utils import parse_string, parse_html_string, guess_type, get_pages_for_items
def _check_deprecated(self):
    if not self.options.get('ignore_ncx'):
        warnings.warn('In the future version we will turn default option ignore_ncx to True.')