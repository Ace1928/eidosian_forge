from openstackclient.common import availability_zone
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
def _build_compute_az_datalist(compute_az, long_datalist=False):
    datalist = ()
    if not long_datalist:
        datalist = (compute_az.name, 'available')
    else:
        for host, services in compute_az.hosts.items():
            for service, state in services.items():
                datalist += (compute_az.name, 'available', '', host, service, 'enabled :-) ' + state['updated_at'])
    return (datalist,)