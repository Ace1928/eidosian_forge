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
def _extended_class_manager(self, class_, factory):
    manager = factory(class_)
    if not isinstance(manager, ClassManager):
        manager = _ClassInstrumentationAdapter(class_, manager)
    if factory != ClassManager and (not self._extended):
        self._extended = True
        _install_instrumented_lookups()
    self._manager_finders[class_] = manager.manager_getter()
    self._state_finders[class_] = manager.state_getter()
    self._dict_finders[class_] = manager.dict_getter()
    return manager