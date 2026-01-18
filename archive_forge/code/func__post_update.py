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
def _post_update(self, state, uowcommit, related, is_m2o_delete=False):
    for x in related:
        if not is_m2o_delete or x is not None:
            uowcommit.register_post_update(state, [r for l, r in self.prop.synchronize_pairs])
            break