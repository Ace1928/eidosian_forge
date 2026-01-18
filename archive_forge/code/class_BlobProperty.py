import datetime
from boto.sdb.db.key import Key
from boto.utils import Password
from boto.sdb.db.query import Query
import re
import boto
import boto.s3.key
from boto.sdb.db.blob import Blob
from boto.compat import six, long_type
class BlobProperty(Property):
    data_type = Blob
    type_name = 'blob'

    def __set__(self, obj, value):
        if value != self.default_value():
            if not isinstance(value, Blob):
                oldb = self.__get__(obj, type(obj))
                id = None
                if oldb:
                    id = oldb.id
                b = Blob(value=value, id=id)
                value = b
        super(BlobProperty, self).__set__(obj, value)