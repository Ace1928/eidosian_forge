import copy
import stevedore
from glance.common import location_strategy
from glance.common.location_strategy import location_order
from glance.common.location_strategy import store_type
from glance.tests.unit import base
def _fake_stevedore_extension_manager(*args, **kwargs):

    def ret():
        return None
    ret.names = lambda: modules
    return ret