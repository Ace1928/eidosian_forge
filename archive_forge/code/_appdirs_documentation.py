import inspect
from typing import cast
import appdirs  # type: ignore[import-untyped]
from twisted.python.compat import currentframe

    Get a data directory for the caller function, or C{moduleName} if given.

    @param moduleName: The module name if you don't wish to have the caller's
        module.

    @returns: A directory for putting data in.
    