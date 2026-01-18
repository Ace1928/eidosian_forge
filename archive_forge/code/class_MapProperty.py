import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class MapProperty(Property):
    data_type = dict
    type_name = 'Map'

    def __init__(self, item_type=str, verbose_name=None, name=None, default=None, **kwds):
        if default is None:
            default = {}
        self.item_type = item_type
        super(MapProperty, self).__init__(verbose_name, name, default=default, required=True, **kwds)

    def validate(self, value):
        value = super(MapProperty, self).validate(value)
        if value is not None:
            if not isinstance(value, dict):
                raise ValueError('Value must of type dict')
        if self.item_type in six.integer_types:
            item_type = six.integer_types
        elif self.item_type in six.string_types:
            item_type = six.string_types
        else:
            item_type = self.item_type
        for key in value:
            if not isinstance(value[key], item_type):
                if item_type == six.integer_types:
                    raise ValueError('Values in the %s Map must all be integers.' % self.name)
                else:
                    raise ValueError('Values in the %s Map must all be %s instances' % (self.name, self.item_type.__name__))
        return value

    def empty(self, value):
        return value is None

    def default_value(self):
        return {}