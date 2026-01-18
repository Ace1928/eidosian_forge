import uuid
import fixtures
from keystoneauth1 import discover
from keystoneauth1 import loading
from keystoneauth1 import plugin
def _format_endpoint(endpoint, **kwargs):
    if kwargs.get('service_type') is plugin.AUTH_INTERFACE:
        kwargs['service_type'] = 'identity'
    version = kwargs.get('version')
    if version:
        discover.normalize_version_number(version)
        kwargs['version'] = '.'.join((str(v) for v in version))
    return endpoint % kwargs