from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _self_href(base_uri, tenant_id, backup_id):
    return '%s/v2/%s/backups/%s' % (base_uri, tenant_id, backup_id)