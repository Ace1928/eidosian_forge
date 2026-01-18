from boto.compat import six
class SelectResultSet(object):

    def __init__(self, domain=None, query='', max_items=None, next_token=None, consistent_read=False):
        self.domain = domain
        self.query = query
        self.consistent_read = consistent_read
        self.max_items = max_items
        self.next_token = next_token

    def __iter__(self):
        more_results = True
        num_results = 0
        while more_results:
            rs = self.domain.connection.select(self.domain, self.query, next_token=self.next_token, consistent_read=self.consistent_read)
            for item in rs:
                if self.max_items and num_results >= self.max_items:
                    raise StopIteration
                yield item
                num_results += 1
            self.next_token = rs.next_token
            if self.max_items and num_results >= self.max_items:
                raise StopIteration
            more_results = self.next_token is not None

    def next(self):
        return next(self.__iter__())