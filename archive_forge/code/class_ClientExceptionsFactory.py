from botocore.exceptions import ClientError
from botocore.utils import get_service_module_name
class ClientExceptionsFactory:

    def __init__(self):
        self._client_exceptions_cache = {}

    def create_client_exceptions(self, service_model):
        """Creates a ClientExceptions object for the particular service client

        :type service_model: botocore.model.ServiceModel
        :param service_model: The service model for the client

        :rtype: object that subclasses from BaseClientExceptions
        :returns: The exceptions object of a client that can be used
            to grab the various different modeled exceptions.
        """
        service_name = service_model.service_name
        if service_name not in self._client_exceptions_cache:
            client_exceptions = self._create_client_exceptions(service_model)
            self._client_exceptions_cache[service_name] = client_exceptions
        return self._client_exceptions_cache[service_name]

    def _create_client_exceptions(self, service_model):
        cls_props = {}
        code_to_exception = {}
        for error_shape in service_model.error_shapes:
            exception_name = str(error_shape.name)
            exception_cls = type(exception_name, (ClientError,), {})
            cls_props[exception_name] = exception_cls
            code = str(error_shape.error_code)
            code_to_exception[code] = exception_cls
        cls_name = str(get_service_module_name(service_model) + 'Exceptions')
        client_exceptions_cls = type(cls_name, (BaseClientExceptions,), cls_props)
        return client_exceptions_cls(code_to_exception)