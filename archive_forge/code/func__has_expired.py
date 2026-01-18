import xml.sax
import datetime
import itertools
from boto import handler
from boto import config
from boto.mturk.price import Price
import boto.mturk.notification
from boto.connection import AWSQueryConnection
from boto.exception import EC2ResponseError
from boto.resultset import ResultSet
from boto.mturk.question import QuestionForm, ExternalQuestion, HTMLQuestion
def _has_expired(self):
    """ Has this HIT expired yet? """
    expired = False
    if hasattr(self, 'Expiration'):
        now = datetime.datetime.utcnow()
        expiration = datetime.datetime.strptime(self.Expiration, '%Y-%m-%dT%H:%M:%SZ')
        expired = now >= expiration
    else:
        raise ValueError('ERROR: Request for expired property, but no Expiration in HIT!')
    return expired