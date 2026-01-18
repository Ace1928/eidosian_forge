import collections
import hashlib
from cliff.formatters import base
class ResourceDotInfo(object):

    def __init__(self, res):
        self.resource = res
        links = {link['rel']: link['href'] for link in res.links}
        self.nested_dot_id = self.dot_id(links.get('nested'), 'stack')
        self.stack_dot_id = self.dot_id(links.get('stack'), 'stack')
        self.res_dot_id = self.dot_id(links.get('self'))

    @staticmethod
    def dot_id(url, prefix=None):
        """Build an id with a prefix and a truncated hash of the URL"""
        if not url:
            return None
        if not prefix:
            prefix = 'r'
        hash_object = hashlib.sha256(url.encode('utf-8'))
        return '%s_%s' % (prefix, hash_object.hexdigest()[:20])