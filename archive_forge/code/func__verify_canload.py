from __future__ import annotations
from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util
def _verify_canload(self, state):
    if self.prop.uselist and state is None:
        raise exc.FlushError("Can't flush None value found in collection %s" % (self.prop,))
    elif state is not None and (not self.mapper._canload(state, allow_subtypes=not self.enable_typechecks)):
        if self.mapper._canload(state, allow_subtypes=True):
            raise exc.FlushError('Attempting to flush an item of type %(x)s as a member of collection "%(y)s". Expected an object of type %(z)s or a polymorphic subclass of this type. If %(x)s is a subclass of %(z)s, configure mapper "%(zm)s" to load this subtype polymorphically, or set enable_typechecks=False to allow any subtype to be accepted for flush. ' % {'x': state.class_, 'y': self.prop, 'z': self.mapper.class_, 'zm': self.mapper})
        else:
            raise exc.FlushError('Attempting to flush an item of type %(x)s as a member of collection "%(y)s". Expected an object of type %(z)s or a polymorphic subclass of this type.' % {'x': state.class_, 'y': self.prop, 'z': self.mapper.class_})