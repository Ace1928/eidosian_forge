import os
import re
import stat
import sys
import fnmatch
from xdg import BaseDirectory
import xdg.Locale
from xml.dom import minidom, XML_NAMESPACE
from collections import defaultdict
class MIMEtype(object):
    """Class holding data about a MIME type.
    
    Calling the class will return a cached instance, so there is only one 
    instance for each MIME type. The name can either be passed as one part
    ('text/plain'), or as two ('text', 'plain').
    """

    def __new__(cls, media, subtype=None):
        if subtype is None and '/' in media:
            media, subtype = media.split('/', 1)
        assert '/' not in subtype
        media = media.lower()
        subtype = subtype.lower()
        try:
            return types[media, subtype]
        except KeyError:
            mtype = super(MIMEtype, cls).__new__(cls)
            mtype._init(media, subtype)
            types[media, subtype] = mtype
            return mtype

    def _init(self, media, subtype):
        self.media = media
        self.subtype = subtype
        self._comment = None

    def _load(self):
        """Loads comment for current language. Use get_comment() instead."""
        resource = os.path.join('mime', self.media, self.subtype + '.xml')
        for path in BaseDirectory.load_data_paths(resource):
            doc = minidom.parse(path)
            if doc is None:
                continue
            for comment in doc.documentElement.getElementsByTagNameNS(FREE_NS, 'comment'):
                lang = comment.getAttributeNS(XML_NAMESPACE, 'lang') or 'en'
                goodness = 1 + (lang in xdg.Locale.langs)
                if goodness > self._comment[0]:
                    self._comment = (goodness, _get_node_data(comment))
                if goodness == 2:
                    return

    def get_comment(self):
        """Returns comment for current language, loading it if needed."""
        if self._comment is None:
            self._comment = (0, str(self))
            self._load()
        return self._comment[1]

    def canonical(self):
        """Returns the canonical MimeType object if this is an alias."""
        update_cache()
        s = str(self)
        if s in aliases:
            return lookup(aliases[s])
        return self

    def inherits_from(self):
        """Returns a set of Mime types which this inherits from."""
        update_cache()
        return set((lookup(t) for t in inheritance[str(self)]))

    def __str__(self):
        return self.media + '/' + self.subtype

    def __repr__(self):
        return 'MIMEtype(%r, %r)' % (self.media, self.subtype)

    def __hash__(self):
        return hash(self.media) ^ hash(self.subtype)