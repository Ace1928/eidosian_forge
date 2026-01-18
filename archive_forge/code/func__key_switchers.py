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
def _key_switchers(self, uow, states):
    switched, notswitched = uow.memo(('pk_switchers', self), lambda: (set(), set()))
    allstates = switched.union(notswitched)
    for s in states:
        if s not in allstates:
            if self._pks_changed(uow, s):
                switched.add(s)
            else:
                notswitched.add(s)
    return switched