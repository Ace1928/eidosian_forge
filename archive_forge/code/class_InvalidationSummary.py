import uuid
from boto.compat import urllib
from boto.resultset import ResultSet
class InvalidationSummary(object):
    """
    Represents InvalidationSummary complex type in CloudFront API that lists
    the id and status of a given invalidation request.
    """

    def __init__(self, connection=None, distribution_id=None, id='', status=''):
        self.connection = connection
        self.distribution_id = distribution_id
        self.id = id
        self.status = status

    def __repr__(self):
        return '<InvalidationSummary: %s>' % self.id

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Id':
            self.id = value
        elif name == 'Status':
            self.status = value

    def get_distribution(self):
        """
        Returns a Distribution object representing the parent CloudFront
        distribution of the invalidation request listed in the
        InvalidationSummary.

        :rtype: :class:`boto.cloudfront.distribution.Distribution`
        :returns: A Distribution object representing the parent CloudFront
                  distribution  of the invalidation request listed in the
                  InvalidationSummary
        """
        return self.connection.get_distribution_info(self.distribution_id)

    def get_invalidation_request(self):
        """
        Returns an InvalidationBatch object representing the invalidation
        request referred to in the InvalidationSummary.

        :rtype: :class:`boto.cloudfront.invalidation.InvalidationBatch`
        :returns: An InvalidationBatch object representing the invalidation
                  request referred to by the InvalidationSummary
        """
        return self.connection.invalidation_request_status(self.distribution_id, self.id)