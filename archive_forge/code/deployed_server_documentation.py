from oslo_config import cfg
from oslo_log import log as logging
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources import server_base
from heat.engine import support
A resource for managing servers that are already deployed.

    A DeployedServer resource manages resources for servers that have been
    deployed externally from OpenStack. These servers can be associated with
    SoftwareDeployments for further orchestration via Heat.
    