import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
def _put_with_retry_for_generation_conflict(self, url, body, resource_provider_uuid, resource_provider_generation=None):
    if resource_provider_generation is None:
        max_tries = GENERATION_CONFLICT_RETRIES
    else:
        max_tries = 1
    body['resource_provider_generation'] = resource_provider_generation
    for i in range(max_tries):
        if resource_provider_generation is None:
            body['resource_provider_generation'] = self.get_resource_provider(resource_provider_uuid=resource_provider_uuid)['generation']
        try:
            return self._put(url, body).json()
        except ks_exc.Conflict as e:
            if e.response.json()['errors'][0]['code'] == 'placement.concurrent_update':
                continue
            raise
    raise n_exc.PlacementResourceProviderGenerationConflict(resource_provider=resource_provider_uuid, generation=body['resource_provider_generation'])