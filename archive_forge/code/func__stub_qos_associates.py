from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_qos_associates(id, name):
    return {'assoications_type': 'volume_type', 'name': name, 'id': id}