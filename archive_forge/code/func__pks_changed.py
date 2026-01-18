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
def _pks_changed(self, uowcommit, state):
    return sync.source_modified(uowcommit, state, self.parent, self.prop.synchronize_pairs)