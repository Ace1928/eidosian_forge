import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class DateTimeProperty(Property):
    """This class handles both the datetime.datetime object
    And the datetime.date objects. It can return either one,
    depending on the value stored in the database"""
    data_type = datetime.datetime
    type_name = 'DateTime'

    def __init__(self, verbose_name=None, auto_now=False, auto_now_add=False, name=None, default=None, required=False, validator=None, choices=None, unique=False):
        super(DateTimeProperty, self).__init__(verbose_name, name, default, required, validator, choices, unique)
        self.auto_now = auto_now
        self.auto_now_add = auto_now_add

    def default_value(self):
        if self.auto_now or self.auto_now_add:
            return self.now()
        return super(DateTimeProperty, self).default_value()

    def validate(self, value):
        if value is None:
            return
        if isinstance(value, datetime.date):
            return value
        return super(DateTimeProperty, self).validate(value)

    def get_value_for_datastore(self, model_instance):
        if self.auto_now:
            setattr(model_instance, self.name, self.now())
        return super(DateTimeProperty, self).get_value_for_datastore(model_instance)

    def now(self):
        return datetime.datetime.utcnow()