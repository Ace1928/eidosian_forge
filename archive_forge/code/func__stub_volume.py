from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _stub_volume(*args, **kwargs):
    volume = {'migration_status': None, 'attachments': [{'server_id': '1234', 'id': '3f88836f-adde-4296-9f6b-2c59a0bcda9a', 'attachment_id': '5678'}], 'links': [{'href': 'http://localhost/v2/fake/volumes/1234', 'rel': 'self'}, {'href': 'http://localhost/fake/volumes/1234', 'rel': 'bookmark'}], 'availability_zone': 'cinder', 'os-vol-host-attr:host': 'ip-192-168-0-2', 'encrypted': 'false', 'updated_at': '2013-11-12T21:00:00.000000', 'os-volume-replication:extended_status': 'None', 'replication_status': 'disabled', 'snapshot_id': None, 'id': 1234, 'size': 1, 'user_id': '1b2d6e8928954ca4ae7c243863404bdc', 'os-vol-tenant-attr:tenant_id': 'eb72eb33a0084acf8eb21356c2b021a7', 'os-vol-mig-status-attr:migstat': None, 'metadata': {}, 'status': 'available', 'description': None, 'os-volume-replication:driver_data': None, 'source_volid': None, 'consistencygroup_id': None, 'os-vol-mig-status-attr:name_id': None, 'name': 'sample-volume', 'bootable': 'false', 'created_at': '2012-08-27T00:00:00.000000', 'volume_type': 'None'}
    volume.update(kwargs)
    return volume