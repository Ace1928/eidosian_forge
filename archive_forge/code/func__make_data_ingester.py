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
def _make_data_ingester(self):
    """Determines the right data ingester, starts it, and returns it."""
    flags = self.flags
    if flags.grpc_data_provider:
        ingester = server_ingester.ExistingServerDataIngester(flags.grpc_data_provider, channel_creds_type=flags.grpc_creds_type)
        ingester.start()
        return ingester
    if flags.load_fast == 'true':
        try:
            return self._start_subprocess_data_ingester()
        except server_ingester.NoDataServerError as e:
            msg = 'Option --load_fast=true not available: %s\n' % e
            sys.stderr.write(msg)
            sys.exit(1)
        except server_ingester.DataServerStartupError as e:
            msg = _DATA_SERVER_STARTUP_ERROR_MESSAGE_TEMPLATE % e
            sys.stderr.write(msg)
            sys.exit(1)
    if flags.load_fast == 'auto' and _should_use_data_server(flags):
        try:
            ingester = self._start_subprocess_data_ingester()
            sys.stderr.write(_DATA_SERVER_ADVISORY_MESSAGE)
            sys.stderr.flush()
            return ingester
        except server_ingester.NoDataServerError as e:
            logger.info('No data server: %s', e)
        except server_ingester.DataServerStartupError as e:
            logger.info('Data server error: %s; falling back to multiplexer', e)
    ingester = local_ingester.LocalDataIngester(flags)
    ingester.start()
    return ingester