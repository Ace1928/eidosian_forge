from oslo_log import log
from keystone import assignment
from keystone.common import cache
from keystone.common import driver_hints
from keystone.common import manager
from keystone.common import provider_api
from keystone.common.resource_options import options as ro_opt
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.i18n import _
from keystone import notifications
from keystone.resource.backends import base
from keystone.token import provider as token_provider
def _include_limits(self, projects):
    """Modify a list of projects to include limit information.

        :param projects: a list of project references including an `id`
        :type projects: list of dictionaries
        """
    for project in projects:
        hints = driver_hints.Hints()
        hints.add_filter('project_id', project['id'])
        limits = PROVIDERS.unified_limit_api.list_limits(hints)
        project['limits'] = limits