from abc import ABCMeta
from abc import abstractmethod
import argparse
import atexit
from collections import defaultdict
import errno
import logging
import mimetypes
import os
import shlex
import signal
import socket
import sys
import threading
import time
import urllib.parse
from absl import flags as absl_flags
from absl.flags import argparse_flags
from werkzeug import serving
from tensorboard import manager
from tensorboard import version
from tensorboard.backend import application
from tensorboard.backend.event_processing import data_ingester as local_ingester
from tensorboard.backend.event_processing import event_file_inspector as efi
from tensorboard.data import server_ingester
from tensorboard.plugins.core import core_plugin
from tensorboard.util import tb_logging
def _get_wildcard_address(self, port):
    """Returns a wildcard address for the port in question.

        This will attempt to follow the best practice of calling
        getaddrinfo() with a null host and AI_PASSIVE to request a
        server-side socket wildcard address. If that succeeds, this
        returns the first IPv6 address found, or if none, then returns
        the first IPv4 address. If that fails, then this returns the
        hardcoded address "::" if socket.has_ipv6 is True, else
        "0.0.0.0".
        """
    fallback_address = '::' if socket.has_ipv6 else '0.0.0.0'
    if hasattr(socket, 'AI_PASSIVE'):
        try:
            addrinfos = socket.getaddrinfo(None, port, socket.AF_UNSPEC, socket.SOCK_STREAM, socket.IPPROTO_TCP, socket.AI_PASSIVE)
        except socket.gaierror as e:
            logger.warning('Failed to auto-detect wildcard address, assuming %s: %s', fallback_address, str(e))
            return fallback_address
        addrs_by_family = defaultdict(list)
        for family, _, _, _, sockaddr in addrinfos:
            addrs_by_family[family].append(sockaddr[0])
        if hasattr(socket, 'AF_INET6') and addrs_by_family[socket.AF_INET6]:
            return addrs_by_family[socket.AF_INET6][0]
        if hasattr(socket, 'AF_INET') and addrs_by_family[socket.AF_INET]:
            return addrs_by_family[socket.AF_INET][0]
    logger.warning('Failed to auto-detect wildcard address, assuming %s', fallback_address)
    return fallback_address