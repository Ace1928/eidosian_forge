from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_consistencygroup(detailed=True, **kwargs):
    consistencygroup = {'name': 'cg', 'id': '11111111-1111-1111-1111-111111111111'}
    if detailed:
        details = {'created_at': '2012-08-28T16:30:31.000000', 'description': None, 'availability_zone': 'myzone', 'status': 'available'}
        consistencygroup.update(details)
    consistencygroup.update(kwargs)
    return consistencygroup