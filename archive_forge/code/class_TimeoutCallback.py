from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
class TimeoutCallback(SessionCallback):
    """ Represent a callback to execute once on the ``IOLoop`` after a specified
    time interval passes.

    """
    _timeout: int

    def __init__(self, callback: Callback, timeout: int, *, callback_id: ID) -> None:
        """

        Args:
            callback (callable) :

            timeout (int) :

            id (ID) :

        """
        super().__init__(callback=callback, callback_id=callback_id)
        self._timeout = timeout

    @property
    def timeout(self) -> int:
        """ The timeout (in milliseconds) that the callback should run after.

        """
        return self._timeout