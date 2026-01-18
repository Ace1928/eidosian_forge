import abc
import contextlib
import enum
import logging
import sys
from grpc import _compression
from grpc._cython import cygrpc as _cygrpc
from grpc._runtime_protos import protos
from grpc._runtime_protos import protos_and_services
from grpc._runtime_protos import services
class ServiceRpcHandler(GenericRpcHandler, metaclass=abc.ABCMeta):
    """An implementation of RPC methods belonging to a service.

    A service handles RPC methods with structured names of the form
    '/Service.Name/Service.Method', where 'Service.Name' is the value
    returned by service_name(), and 'Service.Method' is the method
    name.  A service can have multiple method names, but only a single
    service name.
    """

    @abc.abstractmethod
    def service_name(self):
        """Returns this service's name.

        Returns:
          The service name.
        """
        raise NotImplementedError()