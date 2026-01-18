import logging
import os
from oslo_config import cfg
from oslo_middleware import cors
from oslo_policy import opts
from oslo_policy import policy
from paste import deploy
from glance.i18n import _
from glance.version import version_info as version
def _get_deployment_config_file():
    """
    Retrieve the deployment_config_file config item, formatted as an
    absolute pathname.
    """
    path = CONF.paste_deploy.config_file
    if not path:
        path = _get_paste_config_path()
    if not path or not os.path.isfile(os.path.abspath(path)):
        msg = _('Unable to locate paste config file for %s.') % CONF.prog
        raise RuntimeError(msg)
    return os.path.abspath(path)