from boto.compat import six
class QueryResultSet(object):

    def __init__(self, domain=None, query='', max_items=None, attr_names=None):
        self.max_items = max_items
        self.domain = domain
        self.query = query
        self.attr_names = attr_names

    def __iter__(self):
        return query_lister(self.domain, self.query, self.max_items, self.attr_names)