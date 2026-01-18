import copy
from oslo_config import cfg
from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources import scheduler_hints as sh
def _availability_zone(self):
    """Return Server's Availability Zone.

        Fetching it from Nova if necessary.
        """
    availability_zone = self.properties[self.AVAILABILITY_ZONE]
    if availability_zone is None:
        try:
            server = self.client().servers.get(self.resource_id)
        except Exception as e:
            self.client_plugin().ignore_not_found(e)
            return
    return getattr(server, 'OS-EXT-AZ:availability_zone', None)