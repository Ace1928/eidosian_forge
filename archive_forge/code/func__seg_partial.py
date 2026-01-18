from neutron_lib.api.definitions import multiprovidernet
from neutron_lib.api.definitions import provider_net
from neutron_lib.exceptions import multiprovidernet as mp_exc
from neutron_lib.tests.unit.api.definitions import base
from neutron_lib.tests.unit.api.validators import test_multiprovidernet \
def _seg_partial(seg):
    return seg[provider_net.PHYSICAL_NETWORK] == 'pn0'