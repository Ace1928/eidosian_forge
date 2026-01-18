import logging
import time
import weakref
from botocore import xform_name
from botocore.exceptions import BotoCoreError, ConnectionError, HTTPClientError
from botocore.model import OperationNotFoundError
from botocore.utils import CachedProperty
class EndpointDiscoveryHandler:

    def __init__(self, manager):
        self._manager = manager

    def register(self, events, service_id):
        events.register('before-parameter-build.%s' % service_id, self.gather_identifiers)
        events.register_first('request-created.%s' % service_id, self.discover_endpoint)
        events.register('needs-retry.%s' % service_id, self.handle_retries)

    def gather_identifiers(self, params, model, context, **kwargs):
        endpoint_discovery = model.endpoint_discovery
        if endpoint_discovery is None:
            return
        ids = self._manager.gather_identifiers(model, params)
        context['discovery'] = {'identifiers': ids}

    def discover_endpoint(self, request, operation_name, **kwargs):
        ids = request.context.get('discovery', {}).get('identifiers')
        if ids is None:
            return
        endpoint = self._manager.describe_endpoint(Operation=operation_name, Identifiers=ids)
        if endpoint is None:
            logger.debug('Failed to discover and inject endpoint')
            return
        if not endpoint.startswith('http'):
            endpoint = 'https://' + endpoint
        logger.debug('Injecting discovered endpoint: %s', endpoint)
        request.url = endpoint

    def handle_retries(self, request_dict, response, operation, **kwargs):
        if response is None:
            return None
        _, response = response
        status = response.get('ResponseMetadata', {}).get('HTTPStatusCode')
        error_code = response.get('Error', {}).get('Code')
        if status != 421 and error_code != 'InvalidEndpointException':
            return None
        context = request_dict.get('context', {})
        ids = context.get('discovery', {}).get('identifiers')
        if ids is None:
            return None
        self._manager.delete_endpoints(Operation=operation.name, Identifiers=ids)
        return 0