from __future__ import annotations
from typing import (
from zope.interface import Attribute, Interface
from twisted.python.failure import Failure
class IProducer(Interface):
    """
    A producer produces data for a consumer.

    Typically producing is done by calling the C{write} method of a class
    implementing L{IConsumer}.
    """

    def stopProducing() -> None:
        """
        Stop producing data.

        This tells a producer that its consumer has died, so it must stop
        producing data for good.
        """