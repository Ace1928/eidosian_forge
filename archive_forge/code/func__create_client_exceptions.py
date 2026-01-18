from botocore.exceptions import ClientError
from botocore.utils import get_service_module_name
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