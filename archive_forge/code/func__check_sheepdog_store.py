import sys
import glance_store
from oslo_config import cfg
from oslo_upgradecheck import common_checks
from oslo_upgradecheck import upgradecheck
from glance.common import removed_config
from glance.common import wsgi  # noqa
def _check_sheepdog_store(self):
    """Check that the removed sheepdog backend store is not configured."""
    glance_store.register_opts(CONF)
    sheepdog_present = False
    if 'sheepdog' in (getattr(CONF, 'enabled_backends') or {}):
        sheepdog_present = True
    if 'sheepdog' in (getattr(CONF.glance_store, 'stores') or []):
        sheepdog_present = True
    if sheepdog_present:
        return upgradecheck.Result(FAILURE, 'The "sheepdog" backend store driver has been removed, but current settings have it configured.')
    return upgradecheck.Result(SUCCESS)