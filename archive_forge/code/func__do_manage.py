import collections
import ipaddress
from oslo_utils import uuidutils
import re
import string
from manilaclient import api_versions
from manilaclient import base
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient.v2 import share_instances
def _do_manage(self, service_host, protocol, export_path, driver_options=None, share_type=None, name=None, description=None, is_public=None, share_server_id=None, resource_path='/shares/manage'):
    """Manage some existing share.

        :param service_host: text - host where manila share service is running
        :param protocol: text - share protocol that is used
        :param export_path: text - export path of share
        :param driver_options: dict - custom set of key-values
        :param share_type: text - share type that should be used for share
        :param name: text - name of new share
        :param description: - description for new share
        :param is_public: - visibility for new share
        :param share_server_id: text - id of share server associated with share
        """
    driver_options = driver_options if driver_options else dict()
    body = {'service_host': service_host, 'share_type': share_type, 'protocol': protocol, 'export_path': export_path, 'driver_options': driver_options, 'name': name, 'description': description, 'share_server_id': share_server_id}
    if is_public is not None:
        body['is_public'] = is_public
    return self._create(resource_path, {'share': body}, 'share')