import sys
import pickle
import typing
import binascii
import contextlib
from io import BytesIO
from aiokeydb.types.serializer import BaseSerializer
from pickle import DEFAULT_PROTOCOL, Pickler, Unpickler
class CloudPickleSerializer(BaseSerializer):

    @staticmethod
    def dumps(obj: typing.Any, protocol: int=cloudpickle.DEFAULT_PROTOCOL, *args, **kwargs) -> bytes:
        """
            Dumps an object to bytes
            """
        return cloudpickle.dumps(obj, *args, protocol=protocol, **kwargs)

    @staticmethod
    def loads(data: typing.Union[str, bytes, typing.Any], *args, **kwargs) -> typing.Any:
        """
            Loads an object from bytes
            """
        return cloudpickle.loads(data, *args, **kwargs)

    @staticmethod
    def register_module(module: ModuleType):
        """
            Registers a module with cloudpickle
            """
        cloudpickle.register_pickle_by_value(module)

    @staticmethod
    def unregister_module(module: ModuleType):
        """
            Registers a class with cloudpickle
            """
        cloudpickle.unregister_pickle_by_value(module)