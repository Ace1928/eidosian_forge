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
def _init_play_order(self):
    self._play_order = {'enabled': False, 'start_from': 1}
    try:
        self._play_order['enabled'] = self.options['play_order']['enabled']
        self._play_order['start_from'] = self.options['play_order']['start_from']
    except KeyError:
        pass