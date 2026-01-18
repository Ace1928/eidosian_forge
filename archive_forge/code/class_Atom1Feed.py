import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class Atom1Feed(SyndicationFeed):
    content_type = 'application/atom+xml; charset=utf-8'
    ns = 'http://www.w3.org/2005/Atom'

    def write(self, outfile, encoding):
        handler = SimplerXMLGenerator(outfile, encoding, short_empty_elements=True)
        handler.startDocument()
        handler.startElement('feed', self.root_attributes())
        self.add_root_elements(handler)
        self.write_items(handler)
        handler.endElement('feed')

    def root_attributes(self):
        if self.feed['language'] is not None:
            return {'xmlns': self.ns, 'xml:lang': self.feed['language']}
        else:
            return {'xmlns': self.ns}

    def add_root_elements(self, handler):
        handler.addQuickElement('title', self.feed['title'])
        handler.addQuickElement('link', '', {'rel': 'alternate', 'href': self.feed['link']})
        if self.feed['feed_url'] is not None:
            handler.addQuickElement('link', '', {'rel': 'self', 'href': self.feed['feed_url']})
        handler.addQuickElement('id', self.feed['id'])
        handler.addQuickElement('updated', rfc3339_date(self.latest_post_date()))
        if self.feed['author_name'] is not None:
            handler.startElement('author', {})
            handler.addQuickElement('name', self.feed['author_name'])
            if self.feed['author_email'] is not None:
                handler.addQuickElement('email', self.feed['author_email'])
            if self.feed['author_link'] is not None:
                handler.addQuickElement('uri', self.feed['author_link'])
            handler.endElement('author')
        if self.feed['subtitle'] is not None:
            handler.addQuickElement('subtitle', self.feed['subtitle'])
        for cat in self.feed['categories']:
            handler.addQuickElement('category', '', {'term': cat})
        if self.feed['feed_copyright'] is not None:
            handler.addQuickElement('rights', self.feed['feed_copyright'])

    def write_items(self, handler):
        for item in self.items:
            handler.startElement('entry', self.item_attributes(item))
            self.add_item_elements(handler, item)
            handler.endElement('entry')

    def add_item_elements(self, handler, item):
        handler.addQuickElement('title', item['title'])
        handler.addQuickElement('link', '', {'href': item['link'], 'rel': 'alternate'})
        if item['pubdate'] is not None:
            handler.addQuickElement('published', rfc3339_date(item['pubdate']))
        if item['updateddate'] is not None:
            handler.addQuickElement('updated', rfc3339_date(item['updateddate']))
        if item['author_name'] is not None:
            handler.startElement('author', {})
            handler.addQuickElement('name', item['author_name'])
            if item['author_email'] is not None:
                handler.addQuickElement('email', item['author_email'])
            if item['author_link'] is not None:
                handler.addQuickElement('uri', item['author_link'])
            handler.endElement('author')
        if item['unique_id'] is not None:
            unique_id = item['unique_id']
        else:
            unique_id = get_tag_uri(item['link'], item['pubdate'])
        handler.addQuickElement('id', unique_id)
        if item['description'] is not None:
            handler.addQuickElement('summary', item['description'], {'type': 'html'})
        for enclosure in item['enclosures']:
            handler.addQuickElement('link', '', {'rel': 'enclosure', 'href': enclosure.url, 'length': enclosure.length, 'type': enclosure.mime_type})
        for cat in item['categories']:
            handler.addQuickElement('category', '', {'term': cat})
        if item['item_copyright'] is not None:
            handler.addQuickElement('rights', item['item_copyright'])