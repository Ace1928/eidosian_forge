import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class IntegerProperty(Property):
    data_type = int
    type_name = 'Integer'

    def __init__(self, verbose_name=None, name=None, default=0, required=False, validator=None, choices=None, unique=False, max=2147483647, min=-2147483648):
        super(IntegerProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.max = max
        self.min = min

    def validate(self, value):
        value = int(value)
        value = super(IntegerProperty, self).validate(value)
        if value > self.max:
            raise ValueError('Maximum value is %d' % self.max)
        if value < self.min:
            raise ValueError('Minimum value is %d' % self.min)
        return value

    def empty(self, value):
        return value is None

    def __set__(self, obj, value):
        if value == '' or value is None:
            value = 0
        return super(IntegerProperty, self).__set__(obj, value)