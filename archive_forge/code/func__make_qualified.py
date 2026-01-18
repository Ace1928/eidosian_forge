from boto.route53 import exception
import random
import uuid
import xml.sax
import boto
from boto.connection import AWSAuthConnection
from boto import handler
import boto.jsonresponse
from boto.route53.record import ResourceRecordSets
from boto.route53.zone import Zone
from boto.compat import six, urllib
def _make_qualified(self, value):
    """
        Ensure passed domain names end in a period (.) character.
        This will usually make a domain fully qualified.
        """
    if type(value) in [list, tuple, set]:
        new_list = []
        for record in value:
            if record and (not record[-1] == '.'):
                new_list.append('%s.' % record)
            else:
                new_list.append(record)
        return new_list
    else:
        value = value.strip()
        if value and (not value[-1] == '.'):
            value = '%s.' % value
        return value