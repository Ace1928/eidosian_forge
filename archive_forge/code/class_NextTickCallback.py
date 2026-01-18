from __future__ import annotations
import logging # isort:skip
from typing import TYPE_CHECKING, Callable, Sequence
from ..core.types import ID
from ..util.tornado import _CallbackGroup
class NextTickCallback(SessionCallback):
    """ Represent a callback to execute on the next ``IOLoop`` "tick".

    """

    def __init__(self, callback: Callback, *, callback_id: ID) -> None:
        """

         Args:
            callback (callable) :

            id (ID) :

        """
        super().__init__(callback=callback, callback_id=callback_id)