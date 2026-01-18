import boto
import re
from boto.utils import find_class
import uuid
from boto.sdb.db.key import Key
from boto.sdb.db.blob import Blob
from boto.sdb.db.property import ListProperty, MapProperty
from datetime import datetime, date, time
from boto.exception import SDBPersistenceError, S3ResponseError
from boto.compat import map, six, long_type
def _object_lister(self, cls, query_lister):
    for item in query_lister:
        obj = self.get_object(cls, item.name, item)
        if obj:
            yield obj