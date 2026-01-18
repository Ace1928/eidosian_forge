from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
class CommandListener(_EventListener):
    """Abstract base class for command listeners.

    Handles `CommandStartedEvent`, `CommandSucceededEvent`,
    and `CommandFailedEvent`.
    """

    def started(self, event: CommandStartedEvent) -> None:
        """Abstract method to handle a `CommandStartedEvent`.

        :Parameters:
          - `event`: An instance of :class:`CommandStartedEvent`.
        """
        raise NotImplementedError

    def succeeded(self, event: CommandSucceededEvent) -> None:
        """Abstract method to handle a `CommandSucceededEvent`.

        :Parameters:
          - `event`: An instance of :class:`CommandSucceededEvent`.
        """
        raise NotImplementedError

    def failed(self, event: CommandFailedEvent) -> None:
        """Abstract method to handle a `CommandFailedEvent`.

        :Parameters:
          - `event`: An instance of :class:`CommandFailedEvent`.
        """
        raise NotImplementedError