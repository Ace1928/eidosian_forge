from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_restore():
    return {'volume_id': '712f4980-5ac1-41e5-9383-390aa7c9f58b'}