import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def _handle_existing_loggers(existing, child_loggers, disable_existing):
    """
    When (re)configuring logging, handle loggers which were in the previous
    configuration but are not in the new configuration. There's no point
    deleting them as other threads may continue to hold references to them;
    and by disabling them, you stop them doing any logging.

    However, don't disable children of named loggers, as that's probably not
    what was intended by the user. Also, allow existing loggers to NOT be
    disabled if disable_existing is false.
    """
    root = logging.root
    for log in existing:
        logger = root.manager.loggerDict[log]
        if log in child_loggers:
            if not isinstance(logger, logging.PlaceHolder):
                logger.setLevel(logging.NOTSET)
                logger.handlers = []
                logger.propagate = True
        else:
            logger.disabled = disable_existing