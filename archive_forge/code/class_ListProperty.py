import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class ListProperty(Property):
    data_type = list
    type_name = 'List'

    def __init__(self, item_type, verbose_name=None, name=None, default=None, **kwds):
        if default is None:
            default = []
        self.item_type = item_type
        super(ListProperty, self).__init__(verbose_name, name, default=default, required=True, **kwds)

    def validate(self, value):
        if self.validator:
            self.validator(value)
        if value is not None:
            if not isinstance(value, list):
                value = [value]
        if self.item_type in six.integer_types:
            item_type = six.integer_types
        elif self.item_type in six.string_types:
            item_type = six.string_types
        else:
            item_type = self.item_type
        for item in value:
            if not isinstance(item, item_type):
                if item_type == six.integer_types:
                    raise ValueError('Items in the %s list must all be integers.' % self.name)
                else:
                    raise ValueError('Items in the %s list must all be %s instances' % (self.name, self.item_type.__name__))
        return value

    def empty(self, value):
        return value is None

    def default_value(self):
        return list(super(ListProperty, self).default_value())

    def __set__(self, obj, value):
        """Override the set method to allow them to set the property to an instance of the item_type instead of requiring a list to be passed in"""
        if self.item_type in six.integer_types:
            item_type = six.integer_types
        elif self.item_type in six.string_types:
            item_type = six.string_types
        else:
            item_type = self.item_type
        if isinstance(value, item_type):
            value = [value]
        elif value is None:
            value = []
        return super(ListProperty, self).__set__(obj, value)