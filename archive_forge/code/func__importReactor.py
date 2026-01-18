from twisted.python.log import startLoggingWithObserver
from twisted.python.runtime import platform
from ._shutdown import _watchdog, register
from ._eventloop import (
from ._eventloop import TimeoutError  # pylint: disable=redefined-builtin
from ._version import get_versions
def _importReactor():
    from twisted.internet import reactor
    return reactor