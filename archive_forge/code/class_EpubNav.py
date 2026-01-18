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
class EpubNav(EpubHtml):
    """
    Represents Navigation Document in the EPUB file.
    """

    def __init__(self, uid='nav', file_name='nav.xhtml', media_type='application/xhtml+xml', title=''):
        super(EpubNav, self).__init__(uid=uid, file_name=file_name, media_type=media_type, title=title)

    def is_chapter(self):
        """
        Returns if this document is chapter or not.

        :Returns:
          Returns book value.
        """
        return False

    def __str__(self):
        return '<EpubNav:%s:%s>' % (self.id, self.file_name)