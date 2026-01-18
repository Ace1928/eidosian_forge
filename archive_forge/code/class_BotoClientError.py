import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class BotoClientError(StandardError):
    """
    General Boto Client error (error accessing AWS)
    """

    def __init__(self, reason, *args):
        super(BotoClientError, self).__init__(reason, *args)
        self.reason = reason

    def __repr__(self):
        return 'BotoClientError: %s' % self.reason

    def __str__(self):
        return 'BotoClientError: %s' % self.reason