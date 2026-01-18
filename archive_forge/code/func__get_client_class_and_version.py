import warnings
from oslo_utils import importutils
from zunclient import api_versions
def _get_client_class_and_version(version):
    if not isinstance(version, api_versions.APIVersion):
        version = api_versions.get_api_version(version)
    else:
        api_versions.check_major_version(version)
    return (version, importutils.import_class('zunclient.v%s.client.Client' % version.ver_major))