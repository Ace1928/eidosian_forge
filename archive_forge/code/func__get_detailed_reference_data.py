import copy
from unittest import mock
from osc_lib import exceptions
from openstackclient.common import quota
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
@staticmethod
def _get_detailed_reference_data(quota):
    reference_data = []
    for name, values in quota.to_dict().items():
        if type(values) is dict:
            if 'used' in values:
                in_use = values['used']
            else:
                in_use = values['in_use']
            resource_values = [in_use, values['reserved'], values['limit']]
            reference_data.append(tuple([name] + resource_values))
    return reference_data