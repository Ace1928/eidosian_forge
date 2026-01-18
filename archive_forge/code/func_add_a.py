import copy
from boto.exception import TooManyRecordsException
from boto.route53.record import ResourceRecordSets
from boto.route53.status import Status
def add_a(self, name, value, ttl=None, identifier=None, comment=''):
    """
        Add a new A record to this Zone.  See _new_record for
        parameter documentation.  Returns a Status object.
        """
    ttl = ttl or default_ttl
    name = self.route53connection._make_qualified(name)
    return self.add_record(resource_type='A', name=name, value=value, ttl=ttl, identifier=identifier, comment=comment)