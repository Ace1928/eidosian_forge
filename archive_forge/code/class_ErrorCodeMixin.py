from typing import Dict, Type
from libcloud.utils.py3 import httplib, xmlrpclib
from libcloud.common.base import Response, Connection
from libcloud.common.types import LibcloudError
class ErrorCodeMixin:
    """
    This is a helper for API's that have a well defined collection of error
    codes that are easily parsed out of error messages. It acts as a factory:
    it finds the right exception for the error code, fetches any parameters it
    needs from the context and raises it.
    """
    exceptions = {}

    def raise_exception_for_error(self, error_code, message):
        exceptionCls = self.exceptions.get(error_code, None)
        if exceptionCls is None:
            return
        context = self.connection.context
        driver = self.connection.driver
        params = {}
        if hasattr(exceptionCls, 'kwargs'):
            for key in exceptionCls.kwargs:
                if key in context:
                    params[key] = context[key]
        raise exceptionCls(value=message, driver=driver, **params)