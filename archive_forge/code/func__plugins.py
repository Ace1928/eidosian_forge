from debtcollector import removals
import sqlalchemy
from stevedore import enabled
from oslo_db import exception
@property
def _plugins(self):
    return sorted((ext.obj for ext in self._manager.extensions))