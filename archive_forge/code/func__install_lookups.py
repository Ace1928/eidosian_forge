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
def _install_lookups(lookups):
    global instance_state, instance_dict
    global manager_of_class, opt_manager_of_class
    instance_state = lookups['instance_state']
    instance_dict = lookups['instance_dict']
    manager_of_class = lookups['manager_of_class']
    opt_manager_of_class = lookups['opt_manager_of_class']
    orm_base.instance_state = attributes.instance_state = orm_instrumentation.instance_state = instance_state
    orm_base.instance_dict = attributes.instance_dict = orm_instrumentation.instance_dict = instance_dict
    orm_base.manager_of_class = attributes.manager_of_class = orm_instrumentation.manager_of_class = manager_of_class
    orm_base.opt_manager_of_class = orm_util.opt_manager_of_class = attributes.opt_manager_of_class = orm_instrumentation.opt_manager_of_class = opt_manager_of_class