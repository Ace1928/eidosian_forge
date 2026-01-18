from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
@runtime_checkable
class Serializable(Protocol):
    """Defines methods to serialize and deserialize objects from and into a data stream."""
    __slots__ = ()

    def _serialize(self, stream: 'BytesIO') -> 'Serializable':
        """Serialize the data of this object into the given data stream.

        :note: A serialized object would :meth:`_deserialize` into the same object.

        :param stream: a file-like object

        :return: self
        """
        raise NotImplementedError('To be implemented in subclass')

    def _deserialize(self, stream: 'BytesIO') -> 'Serializable':
        """Deserialize all information regarding this object from the stream.

        :param stream: a file-like object

        :return: self
        """
        raise NotImplementedError('To be implemented in subclass')