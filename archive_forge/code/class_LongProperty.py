import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class LongProperty(Property):
    data_type = long_type
    type_name = 'Long'

    def __init__(self, verbose_name=None, name=None, default=0, required=False, validator=None, choices=None, unique=False):
        super(LongProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)

    def validate(self, value):
        value = long_type(value)
        value = super(LongProperty, self).validate(value)
        min = -9223372036854775808
        max = 9223372036854775807
        if value > max:
            raise ValueError('Maximum value is %d' % max)
        if value < min:
            raise ValueError('Minimum value is %d' % min)
        return value

    def empty(self, value):
        return value is None