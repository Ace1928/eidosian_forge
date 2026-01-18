from oslo_config import cfg
from oslo_log import log as logging
from glance.api import versions
from glance.common import wsgi
def _get_allowed_versions(self):
    allowed_versions = {}
    allowed_versions['v2'] = 2
    allowed_versions['v2.0'] = 2
    allowed_versions['v2.1'] = 2
    allowed_versions['v2.2'] = 2
    allowed_versions['v2.3'] = 2
    allowed_versions['v2.4'] = 2
    allowed_versions['v2.5'] = 2
    allowed_versions['v2.6'] = 2
    allowed_versions['v2.7'] = 2
    allowed_versions['v2.9'] = 2
    if CONF.image_cache_dir:
        allowed_versions['v2.14'] = 2
        allowed_versions['v2.16'] = 2
    allowed_versions['v2.15'] = 2
    if CONF.enabled_backends:
        allowed_versions['v2.8'] = 2
        allowed_versions['v2.10'] = 2
        allowed_versions['v2.11'] = 2
        allowed_versions['v2.12'] = 2
        allowed_versions['v2.13'] = 2
    return allowed_versions