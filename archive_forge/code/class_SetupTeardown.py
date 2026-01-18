import datetime
import logging
import sys
import threading
from typing import TYPE_CHECKING, Any, List, Optional, TypeVar
import psutil
@runtime_checkable
class SetupTeardown(Protocol):
    """Protocol for classes that require setup and teardown."""

    def setup(self) -> None:
        """Extra setup required for the metric beyond __init__."""
        ...

    def teardown(self) -> None:
        """Extra teardown required for the metric."""
        ...