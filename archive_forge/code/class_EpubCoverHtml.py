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
class EpubCoverHtml(EpubHtml):
    """
    Represents Cover page in the EPUB file.
    """

    def __init__(self, uid='cover', file_name='cover.xhtml', image_name='', title='Cover'):
        super(EpubCoverHtml, self).__init__(uid=uid, file_name=file_name, title=title)
        self.image_name = image_name
        self.is_linear = False

    def is_chapter(self):
        """
        Returns if this document is chapter or not.

        :Returns:
          Returns book value.
        """
        return False

    def get_content(self):
        """
        Returns content for cover page as HTML string. Content will be of type 'str' (Python 2) or 'bytes' (Python 3).

        :Returns:
          Returns content of this document.
        """
        self.content = self.book.get_template('cover')
        tree = parse_string(super(EpubCoverHtml, self).get_content())
        tree_root = tree.getroot()
        images = tree_root.xpath('//xhtml:img', namespaces={'xhtml': NAMESPACES['XHTML']})
        images[0].set('src', self.image_name)
        images[0].set('alt', self.title)
        tree_str = etree.tostring(tree, pretty_print=True, encoding='utf-8', xml_declaration=True)
        return tree_str

    def __str__(self):
        return '<EpubCoverHtml:%s:%s>' % (self.id, self.file_name)