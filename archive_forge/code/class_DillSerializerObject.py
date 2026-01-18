import dill
import random
import signal
import anyio
from aiocache import cached
from aiocache.serializers import PickleSerializer, BaseSerializer
from anyio.abc import CancelScope
from typing import List, Union, Callable, Any
from lazyops.utils.logs import logger
class DillSerializerObject(BaseSerializer):
    """
    Transform data to bytes using Dill.dumps and Dill.loads to retrieve it back.
    """
    DEFAULT_ENCODING = None
    PROTOCOL = dill.HIGHEST_PROTOCOL

    def dumps(self, value):
        """
        Serialize the received value using ``Dill.dumps``.

        :param value: obj
        :returns: bytes
        """
        return dill.dumps(value, protocol=self.PROTOCOL)

    def loads(self, value):
        """
        Deserialize value using ``Dill.loads``.

        :param value: bytes
        :returns: obj
        """
        return None if value is None else dill.loads(value)