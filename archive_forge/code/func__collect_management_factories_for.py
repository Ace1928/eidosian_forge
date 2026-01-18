import weakref
from .. import util
from ..orm import attributes
from ..orm import base as orm_base
from ..orm import collections
from ..orm import exc as orm_exc
from ..orm import instrumentation as orm_instrumentation
from ..orm import util as orm_util
from ..orm.instrumentation import _default_dict_getter
from ..orm.instrumentation import _default_manager_getter
from ..orm.instrumentation import _default_opt_manager_getter
from ..orm.instrumentation import _default_state_getter
from ..orm.instrumentation import ClassManager
from ..orm.instrumentation import InstrumentationFactory
def _collect_management_factories_for(self, cls):
    """Return a collection of factories in play or specified for a
        hierarchy.

        Traverses the entire inheritance graph of a cls and returns a
        collection of instrumentation factories for those classes. Factories
        are extracted from active ClassManagers, if available, otherwise
        instrumentation_finders is consulted.

        """
    hierarchy = util.class_hierarchy(cls)
    factories = set()
    for member in hierarchy:
        manager = self.opt_manager_of_class(member)
        if manager is not None:
            factories.add(manager.factory)
        else:
            for finder in instrumentation_finders:
                factory = finder(member)
                if factory is not None:
                    break
            else:
                factory = None
            factories.add(factory)
    factories.discard(None)
    return factories