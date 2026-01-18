from botocore.exceptions import ClientError
from botocore.utils import get_service_module_name
class BaseClientExceptions:
    ClientError = ClientError

    def __init__(self, code_to_exception):
        """Base class for exceptions object on a client

        :type code_to_exception: dict
        :param code_to_exception: Mapping of error codes (strings) to exception
            class that should be raised when encountering a particular
            error code.
        """
        self._code_to_exception = code_to_exception

    def from_code(self, error_code):
        """Retrieves the error class based on the error code

        This is helpful for identifying the exception class needing to be
        caught based on the ClientError.parsed_reponse['Error']['Code'] value

        :type error_code: string
        :param error_code: The error code associated to a ClientError exception

        :rtype: ClientError or a subclass of ClientError
        :returns: The appropriate modeled exception class for that error
            code. If the error code does not match any of the known
            modeled exceptions then return a generic ClientError.
        """
        return self._code_to_exception.get(error_code, self.ClientError)

    def __getattr__(self, name):
        exception_cls_names = [exception_cls.__name__ for exception_cls in self._code_to_exception.values()]
        raise AttributeError(f'{self} object has no attribute {name}. Valid exceptions are: {', '.join(exception_cls_names)}')