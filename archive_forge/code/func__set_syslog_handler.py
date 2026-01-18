import base64
import binascii
import json
import time
import logging
from logging.config import dictConfig
from logging.config import fileConfig
import os
import socket
import sys
import threading
import traceback
from gunicorn import util
def _set_syslog_handler(self, log, cfg, fmt, name):
    prefix = cfg.syslog_prefix or cfg.proc_name.replace(':', '.')
    prefix = 'gunicorn.%s.%s' % (prefix, name)
    fmt = logging.Formatter('%s: %s' % (prefix, fmt))
    try:
        facility = SYSLOG_FACILITIES[cfg.syslog_facility.lower()]
    except KeyError:
        raise RuntimeError('unknown facility name')
    socktype, addr = parse_syslog_address(cfg.syslog_addr)
    h = logging.handlers.SysLogHandler(address=addr, facility=facility, socktype=socktype)
    h.setFormatter(fmt)
    h._gunicorn = True
    log.addHandler(h)