import logging
from osc_lib import utils
from manilaclient import api_versions
from manilaclient import client
from manilaclient.common import constants
from manilaclient import exceptions
def _get_manila_url_from_service_catalog(instance):
    service_type = constants.SFS_SERVICE_TYPE
    url = instance.get_endpoint_for_service_type(constants.SFS_SERVICE_TYPE, region_name=instance._region_name, interface=instance.interface)
    if not url:
        url = instance.get_endpoint_for_service_type(constants.V2_SERVICE_TYPE, region_name=instance._region_name, interface=instance.interface)
        service_type = constants.V2_SERVICE_TYPE
    if url is None:
        raise exceptions.EndpointNotFound(message='Could not find manila / shared-file-system endpoint in the service catalog.')
    return (service_type, url)