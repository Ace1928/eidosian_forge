import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class BooleanProperty(Property):
    data_type = bool
    type_name = 'Boolean'

    def __init__(self, verbose_name=None, name=None, default=False, required=False, validator=None, choices=None, unique=False):
        super(BooleanProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)

    def empty(self, value):
        return value is None