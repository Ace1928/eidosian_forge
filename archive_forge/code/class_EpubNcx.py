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
class EpubNcx(EpubItem):
    """Represents Navigation Control File (NCX) in the EPUB."""

    def __init__(self, uid='ncx', file_name='toc.ncx'):
        super(EpubNcx, self).__init__(uid=uid, file_name=file_name, media_type='application/x-dtbncx+xml')

    def __str__(self):
        return '<EpubNcx:%s>' % self.id