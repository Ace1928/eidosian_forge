import argparse
import getpass
import io
import json
import logging
import signal
import socket
import warnings
from os import environ, walk, _exit as os_exit
from os.path import isfile, isdir, join
from urllib.parse import unquote, urlparse
from sys import argv as sys_argv, exit, stderr, stdin
from time import gmtime, strftime
from swiftclient import RequestException
from swiftclient.utils import config_true_value, generate_temp_url, \
from swiftclient.multithreading import OutputManager
from swiftclient.exceptions import ClientException
from swiftclient import __version__ as client_version
from swiftclient.client import logger_settings as client_logger_settings, \
from swiftclient.service import SwiftService, SwiftError, \
from swiftclient.command_helpers import print_account_stats, \
def _print_compo_cap(name, capabilities):
    for feature, options in sorted(capabilities.items(), key=lambda x: x[0]):
        output_manager.print_msg('%s: %s' % (name, feature))
        if options:
            output_manager.print_msg(' Options:')
            for key, value in sorted(options.items(), key=lambda x: x[0]):
                output_manager.print_msg('  %s: %s' % (key, value))