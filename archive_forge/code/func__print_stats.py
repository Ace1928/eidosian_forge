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
def _print_stats(options, stats, human, totals):
    container = stats.get('container', None)
    for item in stats['listing']:
        item_name = item.get('name')
        if not options['long'] and (not human) and (not options['versions']):
            output_manager.print_msg(item.get('name', item.get('subdir')))
        else:
            if not container:
                item_bytes = item.get('bytes')
                byte_str = prt_bytes(item_bytes, human)
                count = item.get('count')
                totals['count'] += count
                try:
                    meta = item.get('meta')
                    utc = gmtime(float(meta.get('x-timestamp')))
                    datestamp = strftime('%Y-%m-%d %H:%M:%S', utc)
                except TypeError:
                    datestamp = '????-??-?? ??:??:??'
                storage_policy = meta.get('x-storage-policy', '???')
                if not options['totals']:
                    output_manager.print_msg('%12s %s %s %-15s %s', count, byte_str, datestamp, storage_policy, item_name)
            else:
                subdir = item.get('subdir')
                content_type = item.get('content_type')
                if subdir is None:
                    item_bytes = item.get('bytes')
                    byte_str = prt_bytes(item_bytes, human)
                    date, xtime = item.get('last_modified').split('T')
                    xtime = xtime.split('.')[0]
                else:
                    item_bytes = 0
                    byte_str = prt_bytes(item_bytes, human)
                    date = xtime = ''
                    item_name = subdir
                if not options['totals']:
                    if options['versions']:
                        output_manager.print_msg('%s %10s %8s %16s %24s %s', byte_str, date, xtime, item.get('version_id', 'null'), content_type, item_name)
                    else:
                        output_manager.print_msg('%s %10s %8s %24s %s', byte_str, date, xtime, content_type, item_name)
            totals['bytes'] += item_bytes