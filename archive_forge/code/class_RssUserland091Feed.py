import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class RssUserland091Feed(RssFeed):
    _version = '0.91'

    def add_item_elements(self, handler, item):
        handler.addQuickElement('title', item['title'])
        handler.addQuickElement('link', item['link'])
        if item['description'] is not None:
            handler.addQuickElement('description', item['description'])