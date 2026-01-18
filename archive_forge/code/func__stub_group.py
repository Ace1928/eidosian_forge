from datetime import datetime
from cinderclient.tests.unit import fakes
from cinderclient.tests.unit.v3 import fakes_base
from cinderclient.v3 import client
def _stub_group(detailed=True, **kwargs):
    group = {'name': 'test-1', 'id': '1234'}
    if detailed:
        details = {'created_at': '2012-08-28T16:30:31.000000', 'description': 'test-1-desc', 'availability_zone': 'zone1', 'status': 'available', 'group_type': 'my_group_type'}
        group.update(details)
    group.update(kwargs)
    return group