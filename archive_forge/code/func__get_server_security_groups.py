import base64
import functools
import operator
import time
import iso8601
from openstack.cloud import _utils
from openstack.cloud import exc
from openstack.cloud import meta
from openstack.compute.v2._proxy import Proxy
from openstack.compute.v2 import quota_set as _qs
from openstack.compute.v2 import server as _server
from openstack import exceptions
from openstack import utils
def _get_server_security_groups(self, server, security_groups):
    if not self._has_secgroups():
        raise exc.OpenStackCloudUnavailableFeature('Unavailable feature: security groups')
    if not isinstance(server, dict):
        server = self.get_server(server, bare=True)
        if server is None:
            self.log.debug('Server %s not found', server)
            return (None, None)
    if not isinstance(security_groups, (list, tuple)):
        security_groups = [security_groups]
    sec_group_objs = []
    for sg in security_groups:
        if not isinstance(sg, dict):
            sg = self.get_security_group(sg)
            if sg is None:
                self.log.debug('Security group %s not found for adding', sg)
                return (None, None)
        sec_group_objs.append(sg)
    return (server, sec_group_objs)