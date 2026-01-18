from urllib import parse
from keystoneclient.contrib.ec2 import utils as ec2_utils
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.common import password_gen
from heat.engine.clients.os import swift
from heat.engine.resources import stack_user
def _get_heat_signal_url(self, project_id=None):
    """Return a heat-api signal URL for this resource.

        This URL is not pre-signed, valid user credentials are required.
        If a project_id is provided, it is used in place of the original
        project_id. This is useful to generate a signal URL that uses
        the heat stack user project instead of the user's.
        """
    stored = self.data().get('heat_signal_url')
    if stored is not None:
        return stored
    if self.id is None:
        return
    url = self.client_plugin('heat').get_heat_url()
    path = self.identifier().url_path()
    if project_id is not None:
        path = project_id + path[path.find('/'):]
    url = parse.urljoin(url, '%s/signal' % path)
    self.data_set('heat_signal_url', url)
    return url