import operator
import os
import time
import uuid
from keystoneauth1 import discover
import openstack.config
from openstack import connection
from openstack.tests import base
def _pick_flavor(self):
    """Pick a sensible flavor to run tests with.

        This returns None if the compute service is not present (e.g.
        ironic-only deployments).
        """
    if not self.user_cloud.has_service('compute'):
        return None
    flavors = self.user_cloud.list_flavors(get_extra=False)
    flavor_name = os.environ.get('OPENSTACKSDK_FLAVOR')
    if not flavor_name:
        flavor_name = _get_resource_value('flavor_name')
    if flavor_name:
        for flavor in flavors:
            if flavor.name == flavor_name:
                return flavor
        raise self.failureException("Cloud does not have flavor '%s'", flavor_name)
    for flavor in sorted(flavors, key=operator.attrgetter('ram')):
        if 'performance' in flavor.name:
            return flavor
    for flavor in sorted(flavors, key=operator.attrgetter('ram')):
        if flavor.disk:
            return flavor
    raise self.failureException('No sensible flavor found')