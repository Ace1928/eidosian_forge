import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class Rss201rev2Feed(RssFeed):
    _version = '2.0'

    def add_item_elements(self, handler, item):
        handler.addQuickElement('title', item['title'])
        handler.addQuickElement('link', item['link'])
        if item['description'] is not None:
            handler.addQuickElement('description', item['description'])
        if item['author_name'] and item['author_email']:
            handler.addQuickElement('author', '%s (%s)' % (item['author_email'], item['author_name']))
        elif item['author_email']:
            handler.addQuickElement('author', item['author_email'])
        elif item['author_name']:
            handler.addQuickElement('dc:creator', item['author_name'], {'xmlns:dc': 'http://purl.org/dc/elements/1.1/'})
        if item['pubdate'] is not None:
            handler.addQuickElement('pubDate', rfc2822_date(item['pubdate']))
        if item['comments'] is not None:
            handler.addQuickElement('comments', item['comments'])
        if item['unique_id'] is not None:
            guid_attrs = {}
            if isinstance(item.get('unique_id_is_permalink'), bool):
                guid_attrs['isPermaLink'] = str(item['unique_id_is_permalink']).lower()
            handler.addQuickElement('guid', item['unique_id'], guid_attrs)
        if item['ttl'] is not None:
            handler.addQuickElement('ttl', item['ttl'])
        if item['enclosures']:
            enclosures = list(item['enclosures'])
            if len(enclosures) > 1:
                raise ValueError('RSS feed items may only have one enclosure, see http://www.rssboard.org/rss-profile#element-channel-item-enclosure')
            enclosure = enclosures[0]
            handler.addQuickElement('enclosure', '', {'url': enclosure.url, 'length': enclosure.length, 'type': enclosure.mime_type})
        for cat in item['categories']:
            handler.addQuickElement('category', cat)