import difflib
import sys
import six
from apitools.base.protorpclite import messages
from apitools.base.py import base_api
from apitools.base.py import encoding
from apitools.base.py import exceptions
def Mock(self):
    """Stub out the client class with mocked services."""
    client = self.__real_client or self.__client_class(get_credentials=False)

    class Patched(self.__class__, self.__client_class):
        pass
    self.__class__ = Patched
    for name in dir(self.__client_class):
        service_class = getattr(self.__client_class, name)
        if not isinstance(service_class, type):
            continue
        if not issubclass(service_class, base_api.BaseApiService):
            continue
        self.__real_service_classes[name] = service_class
        collection_name = service_class._NAME
        api_name = '%s_%s' % (self.__client_class._PACKAGE, self.__client_class._URL_VERSION)
        mocked_service_class = _MakeMockedService(api_name, collection_name, self, service_class, service_class(client) if self.__real_client else None)
        setattr(self.__client_class, name, mocked_service_class)
        setattr(self, collection_name, mocked_service_class(self))
    self.__real_include_fields = self.__client_class.IncludeFields
    self.__client_class.IncludeFields = self.IncludeFields
    self._url = client._url
    self._http = client._http
    return self