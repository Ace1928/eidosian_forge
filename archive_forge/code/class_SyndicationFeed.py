import datetime
import email
from io import StringIO
from urllib.parse import urlparse
from django.utils.encoding import iri_to_uri
from django.utils.xmlutils import SimplerXMLGenerator
class SyndicationFeed:
    """Base class for all syndication feeds. Subclasses should provide write()"""

    def __init__(self, title, link, description, language=None, author_email=None, author_name=None, author_link=None, subtitle=None, categories=None, feed_url=None, feed_copyright=None, feed_guid=None, ttl=None, **kwargs):

        def to_str(s):
            return str(s) if s is not None else s
        categories = categories and [str(c) for c in categories]
        self.feed = {'title': to_str(title), 'link': iri_to_uri(link), 'description': to_str(description), 'language': to_str(language), 'author_email': to_str(author_email), 'author_name': to_str(author_name), 'author_link': iri_to_uri(author_link), 'subtitle': to_str(subtitle), 'categories': categories or (), 'feed_url': iri_to_uri(feed_url), 'feed_copyright': to_str(feed_copyright), 'id': feed_guid or link, 'ttl': to_str(ttl), **kwargs}
        self.items = []

    def add_item(self, title, link, description, author_email=None, author_name=None, author_link=None, pubdate=None, comments=None, unique_id=None, unique_id_is_permalink=None, categories=(), item_copyright=None, ttl=None, updateddate=None, enclosures=None, **kwargs):
        """
        Add an item to the feed. All args are expected to be strings except
        pubdate and updateddate, which are datetime.datetime objects, and
        enclosures, which is an iterable of instances of the Enclosure class.
        """

        def to_str(s):
            return str(s) if s is not None else s
        categories = categories and [to_str(c) for c in categories]
        self.items.append({'title': to_str(title), 'link': iri_to_uri(link), 'description': to_str(description), 'author_email': to_str(author_email), 'author_name': to_str(author_name), 'author_link': iri_to_uri(author_link), 'pubdate': pubdate, 'updateddate': updateddate, 'comments': to_str(comments), 'unique_id': to_str(unique_id), 'unique_id_is_permalink': unique_id_is_permalink, 'enclosures': enclosures or (), 'categories': categories or (), 'item_copyright': to_str(item_copyright), 'ttl': to_str(ttl), **kwargs})

    def num_items(self):
        return len(self.items)

    def root_attributes(self):
        """
        Return extra attributes to place on the root (i.e. feed/channel) element.
        Called from write().
        """
        return {}

    def add_root_elements(self, handler):
        """
        Add elements in the root (i.e. feed/channel) element. Called
        from write().
        """
        pass

    def item_attributes(self, item):
        """
        Return extra attributes to place on each item (i.e. item/entry) element.
        """
        return {}

    def add_item_elements(self, handler, item):
        """
        Add elements on each item (i.e. item/entry) element.
        """
        pass

    def write(self, outfile, encoding):
        """
        Output the feed in the given encoding to outfile, which is a file-like
        object. Subclasses should override this.
        """
        raise NotImplementedError('subclasses of SyndicationFeed must provide a write() method')

    def writeString(self, encoding):
        """
        Return the feed in the given encoding as a string.
        """
        s = StringIO()
        self.write(s, encoding)
        return s.getvalue()

    def latest_post_date(self):
        """
        Return the latest item's pubdate or updateddate. If no items
        have either of these attributes this return the current UTC date/time.
        """
        latest_date = None
        date_keys = ('updateddate', 'pubdate')
        for item in self.items:
            for date_key in date_keys:
                item_date = item.get(date_key)
                if item_date:
                    if latest_date is None or item_date > latest_date:
                        latest_date = item_date
        return latest_date or datetime.datetime.now(tz=datetime.timezone.utc)