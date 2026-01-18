from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
def _get_raw_item(self):
    return self._manager.get_raw_item(self)