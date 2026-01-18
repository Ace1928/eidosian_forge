from boto.sdb.db.property import Property
from boto.sdb.db.key import Key
from boto.sdb.db.query import Query
import boto
from boto.compat import filter
class Expando(Model):

    def __setattr__(self, name, value):
        if name in self._prop_names:
            object.__setattr__(self, name, value)
        elif name.startswith('_'):
            object.__setattr__(self, name, value)
        elif name == 'id':
            object.__setattr__(self, name, value)
        else:
            self._manager.set_key_value(self, name, value)
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if not name.startswith('_'):
            value = self._manager.get_key_value(self, name)
            if value:
                object.__setattr__(self, name, value)
                return value
        raise AttributeError