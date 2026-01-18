from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
@classmethod
def _get_by_id(cls, id, manager=None):
    if not manager:
        manager = cls._manager
    return manager.get_object(cls, id)