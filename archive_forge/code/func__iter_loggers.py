import logging
from time import monotonic as monotonic_clock
def _iter_loggers():
    """Iterate on existing loggers."""
    yield logging.getLogger()
    manager = logging.Logger.manager
    for logger in manager.loggerDict.values():
        if isinstance(logger, logging.PlaceHolder):
            continue
        yield logger