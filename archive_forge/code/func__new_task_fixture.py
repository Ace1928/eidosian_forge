import http.client
import eventlet
from oslo_serialization import jsonutils as json
from glance.api.v2 import tasks
from glance.common import timeutils
from glance.tests.integration.v2 import base
def _new_task_fixture(**kwargs):
    task_data = {'type': 'import', 'input': {'import_from': 'http://example.com', 'import_from_format': 'qcow2', 'image_properties': {'disk_format': 'vhd', 'container_format': 'ovf'}}}
    task_data.update(kwargs)
    return task_data