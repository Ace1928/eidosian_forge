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
def _install_loggers(cp, handlers, disable_existing):
    """Create and install loggers"""
    llist = cp['loggers']['keys']
    llist = llist.split(',')
    llist = list(_strip_spaces(llist))
    llist.remove('root')
    section = cp['logger_root']
    root = logging.root
    log = root
    if 'level' in section:
        level = section['level']
        log.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)
    hlist = section['handlers']
    if len(hlist):
        hlist = hlist.split(',')
        hlist = _strip_spaces(hlist)
        for hand in hlist:
            log.addHandler(handlers[hand])
    existing = list(root.manager.loggerDict.keys())
    existing.sort()
    child_loggers = []
    for log in llist:
        section = cp['logger_%s' % log]
        qn = section['qualname']
        propagate = section.getint('propagate', fallback=1)
        logger = logging.getLogger(qn)
        if qn in existing:
            i = existing.index(qn) + 1
            prefixed = qn + '.'
            pflen = len(prefixed)
            num_existing = len(existing)
            while i < num_existing:
                if existing[i][:pflen] == prefixed:
                    child_loggers.append(existing[i])
                i += 1
            existing.remove(qn)
        if 'level' in section:
            level = section['level']
            logger.setLevel(level)
        for h in logger.handlers[:]:
            logger.removeHandler(h)
        logger.propagate = propagate
        logger.disabled = 0
        hlist = section['handlers']
        if len(hlist):
            hlist = hlist.split(',')
            hlist = _strip_spaces(hlist)
            for hand in hlist:
                logger.addHandler(handlers[hand])
    _handle_existing_loggers(existing, child_loggers, disable_existing)