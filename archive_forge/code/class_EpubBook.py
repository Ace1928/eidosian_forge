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
class EpubBook(object):

    def __init__(self):
        self.EPUB_VERSION = None
        self.reset()

    def reset(self):
        """Initialises all needed variables to default values"""
        self.metadata = {}
        self.items = []
        self.spine = []
        self.guide = []
        self.pages = []
        self.toc = []
        self.bindings = []
        self.IDENTIFIER_ID = 'id'
        self.FOLDER_NAME = 'EPUB'
        self._id_html = 0
        self._id_image = 0
        self._id_static = 0
        self.title = ''
        self.language = 'en'
        self.direction = None
        self.templates = {'ncx': NCX_XML, 'nav': NAV_XML, 'chapter': CHAPTER_XML, 'cover': COVER_XML}
        self.add_metadata('OPF', 'generator', '', {'name': 'generator', 'content': 'Ebook-lib %s' % '.'.join([str(s) for s in VERSION])})
        self.set_identifier(str(uuid.uuid4()))
        self.prefixes = []
        self.namespaces = {}

    def set_identifier(self, uid):
        """
        Sets unique id for this epub

        :Args:
          - uid: Value of unique identifier for this book
        """
        self.uid = uid
        self.set_unique_metadata('DC', 'identifier', self.uid, {'id': self.IDENTIFIER_ID})

    def set_title(self, title):
        """
        Set title. You can set multiple titles.

        :Args:
          - title: Title value
        """
        self.title = title
        self.add_metadata('DC', 'title', self.title)

    def set_language(self, lang):
        """
        Set language for this epub. You can set multiple languages. Specific items in the book can have
        different language settings.

        :Args:
          - lang: Language code
        """
        self.language = lang
        self.add_metadata('DC', 'language', lang)

    def set_direction(self, direction):
        """
        :Args:
          - direction: Options are "ltr", "rtl" and "default"
        """
        self.direction = direction

    def set_cover(self, file_name, content, create_page=True):
        """
        Set cover and create cover document if needed.

        :Args:
          - file_name: file name of the cover page
          - content: Content for the cover image
          - create_page: Should cover page be defined. Defined as bool value (optional). Default value is True.
        """
        c0 = EpubCover(file_name=file_name)
        c0.content = content
        self.add_item(c0)
        if create_page:
            c1 = EpubCoverHtml(image_name=file_name)
            self.add_item(c1)
        self.add_metadata(None, 'meta', '', OrderedDict([('name', 'cover'), ('content', 'cover-img')]))

    def add_author(self, author, file_as=None, role=None, uid='creator'):
        """Add author for this document"""
        self.add_metadata('DC', 'creator', author, {'id': uid})
        if file_as:
            self.add_metadata(None, 'meta', file_as, {'refines': '#' + uid, 'property': 'file-as', 'scheme': 'marc:relators'})
        if role:
            self.add_metadata(None, 'meta', role, {'refines': '#' + uid, 'property': 'role', 'scheme': 'marc:relators'})

    def add_metadata(self, namespace, name, value, others=None):
        """Add metadata"""
        if namespace in NAMESPACES:
            namespace = NAMESPACES[namespace]
        if namespace not in self.metadata:
            self.metadata[namespace] = {}
        if name not in self.metadata[namespace]:
            self.metadata[namespace][name] = []
        self.metadata[namespace][name].append((value, others))

    def get_metadata(self, namespace, name):
        """Retrieve metadata"""
        if namespace in NAMESPACES:
            namespace = NAMESPACES[namespace]
        return self.metadata[namespace].get(name, [])

    def set_unique_metadata(self, namespace, name, value, others=None):
        """Add metadata if metadata with this identifier does not already exist, otherwise update existing metadata."""
        if namespace in NAMESPACES:
            namespace = NAMESPACES[namespace]
        if namespace in self.metadata and name in self.metadata[namespace]:
            self.metadata[namespace][name] = [(value, others)]
        else:
            self.add_metadata(namespace, name, value, others)

    def add_item(self, item):
        """
        Add additional item to the book. If not defined, media type and chapter id will be defined
        for the item.

        :Args:
          - item: Item instance
        """
        if item.media_type == '':
            has_guessed, media_type = guess_type(item.get_name().lower())
            if has_guessed:
                if media_type is not None:
                    item.media_type = media_type
                else:
                    item.media_type = has_guessed
            else:
                item.media_type = 'application/octet-stream'
        if not item.get_id():
            if isinstance(item, EpubHtml):
                item.id = 'chapter_%d' % self._id_html
                self._id_html += 1
                self.pages += item.pages
            elif isinstance(item, EpubImage):
                item.id = 'image_%d' % self._id_image
                self._id_image += 1
            else:
                item.id = 'static_%d' % self._id_static
                self._id_static += 1
        item.book = self
        self.items.append(item)
        return item

    def get_item_with_id(self, uid):
        """
        Returns item for defined UID.

        >>> book.get_item_with_id('image_001')

        :Args:
          - uid: UID for the item

        :Returns:
          Returns item object. Returns None if nothing was found.
        """
        for item in self.get_items():
            if item.id == uid:
                return item
        return None

    def get_item_with_href(self, href):
        """
        Returns item for defined HREF.

        >>> book.get_item_with_href('EPUB/document.xhtml')

        :Args:
          - href: HREF for the item we are searching for

        :Returns:
          Returns item object. Returns None if nothing was found.
        """
        for item in self.get_items():
            if item.get_name() == href:
                return item
        return None

    def get_items(self):
        """
        Returns all items attached to this book.

        :Returns:
          Returns all items as tuple.
        """
        return (item for item in self.items)

    def get_items_of_type(self, item_type):
        """
        Returns all items of specified type.

        >>> book.get_items_of_type(epub.ITEM_IMAGE)

        :Args:
          - item_type: Type for items we are searching for

        :Returns:
          Returns found items as tuple.
        """
        return (item for item in self.items if item.get_type() == item_type)

    def get_items_of_media_type(self, media_type):
        """
        Returns all items of specified media type.

        :Args:
          - media_type: Media type for items we are searching for

        :Returns:
          Returns found items as tuple.
        """
        return (item for item in self.items if item.media_type == media_type)

    def set_template(self, name, value):
        """
        Defines templates which are used to generate certain types of pages. When defining new value for the template
        we have to use content of type 'str' (Python 2) or 'bytes' (Python 3).

        At the moment we use these templates:
          - ncx
          - nav
          - chapter
          - cover

        :Args:
          - name: Name for the template
          - value: Content for the template
        """
        self.templates[name] = value

    def get_template(self, name):
        """
        Returns value for the template.

        :Args:
          - name: template name

        :Returns:
          Value of the template.
        """
        return self.templates.get(name)

    def add_prefix(self, name, uri):
        """
        Appends custom prefix to be added to the content.opf document

        >>> epub_book.add_prefix('bkterms', 'http://booktype.org/')

        :Args:
          - name: namespave name
          - uri: URI for the namespace
        """
        self.prefixes.append('%s: %s' % (name, uri))