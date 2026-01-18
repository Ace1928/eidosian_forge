from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
class OperationModel:

    def __init__(self, operation_model, service_model, name=None):
        """

        :type operation_model: dict
        :param operation_model: The operation model.  This comes from the
            service model, and is the value associated with the operation
            name in the service model (i.e ``model['operations'][op_name]``).

        :type service_model: botocore.model.ServiceModel
        :param service_model: The service model associated with the operation.

        :type name: string
        :param name: The operation name.  This is the operation name exposed to
            the users of this model.  This can potentially be different from
            the "wire_name", which is the operation name that *must* by
            provided over the wire.  For example, given::

               "CreateCloudFrontOriginAccessIdentity":{
                 "name":"CreateCloudFrontOriginAccessIdentity2014_11_06",
                  ...
              }

           The ``name`` would be ``CreateCloudFrontOriginAccessIdentity``,
           but the ``self.wire_name`` would be
           ``CreateCloudFrontOriginAccessIdentity2014_11_06``, which is the
           value we must send in the corresponding HTTP request.

        """
        self._operation_model = operation_model
        self._service_model = service_model
        self._api_name = name
        self._wire_name = operation_model.get('name')
        self.metadata = service_model.metadata
        self.http = operation_model.get('http', {})

    @CachedProperty
    def name(self):
        if self._api_name is not None:
            return self._api_name
        else:
            return self.wire_name

    @property
    def wire_name(self):
        """The wire name of the operation.

        In many situations this is the same value as the
        ``name``, value, but in some services, the operation name
        exposed to the user is different from the operation name
        we send across the wire (e.g cloudfront).

        Any serialization code should use ``wire_name``.

        """
        return self._operation_model.get('name')

    @property
    def service_model(self):
        return self._service_model

    @CachedProperty
    def documentation(self):
        return self._operation_model.get('documentation', '')

    @CachedProperty
    def deprecated(self):
        return self._operation_model.get('deprecated', False)

    @CachedProperty
    def endpoint_discovery(self):
        return self._operation_model.get('endpointdiscovery', None)

    @CachedProperty
    def is_endpoint_discovery_operation(self):
        return self._operation_model.get('endpointoperation', False)

    @CachedProperty
    def input_shape(self):
        if 'input' not in self._operation_model:
            return None
        return self._service_model.resolve_shape_ref(self._operation_model['input'])

    @CachedProperty
    def output_shape(self):
        if 'output' not in self._operation_model:
            return None
        return self._service_model.resolve_shape_ref(self._operation_model['output'])

    @CachedProperty
    def idempotent_members(self):
        input_shape = self.input_shape
        if not input_shape:
            return []
        return [name for name, shape in input_shape.members.items() if 'idempotencyToken' in shape.metadata and shape.metadata['idempotencyToken']]

    @CachedProperty
    def static_context_parameters(self):
        params = self._operation_model.get('staticContextParams', {})
        return [StaticContextParameter(name=name, value=props.get('value')) for name, props in params.items()]

    @CachedProperty
    def context_parameters(self):
        if not self.input_shape:
            return []
        return [ContextParameter(name=shape.metadata['contextParam']['name'], member_name=name) for name, shape in self.input_shape.members.items() if 'contextParam' in shape.metadata and 'name' in shape.metadata['contextParam']]

    @CachedProperty
    def request_compression(self):
        return self._operation_model.get('requestcompression')

    @CachedProperty
    def auth_type(self):
        return self._operation_model.get('authtype')

    @CachedProperty
    def error_shapes(self):
        shapes = self._operation_model.get('errors', [])
        return list((self._service_model.resolve_shape_ref(s) for s in shapes))

    @CachedProperty
    def endpoint(self):
        return self._operation_model.get('endpoint')

    @CachedProperty
    def http_checksum_required(self):
        return self._operation_model.get('httpChecksumRequired', False)

    @CachedProperty
    def http_checksum(self):
        return self._operation_model.get('httpChecksum', {})

    @CachedProperty
    def has_event_stream_input(self):
        return self.get_event_stream_input() is not None

    @CachedProperty
    def has_event_stream_output(self):
        return self.get_event_stream_output() is not None

    def get_event_stream_input(self):
        return self._get_event_stream(self.input_shape)

    def get_event_stream_output(self):
        return self._get_event_stream(self.output_shape)

    def _get_event_stream(self, shape):
        """Returns the event stream member's shape if any or None otherwise."""
        if shape is None:
            return None
        event_name = shape.event_stream_name
        if event_name:
            return shape.members[event_name]
        return None

    @CachedProperty
    def has_streaming_input(self):
        return self.get_streaming_input() is not None

    @CachedProperty
    def has_streaming_output(self):
        return self.get_streaming_output() is not None

    def get_streaming_input(self):
        return self._get_streaming_body(self.input_shape)

    def get_streaming_output(self):
        return self._get_streaming_body(self.output_shape)

    def _get_streaming_body(self, shape):
        """Returns the streaming member's shape if any; or None otherwise."""
        if shape is None:
            return None
        payload = shape.serialization.get('payload')
        if payload is not None:
            payload_shape = shape.members[payload]
            if payload_shape.type_name == 'blob':
                return payload_shape
        return None

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name})'