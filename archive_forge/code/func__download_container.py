import logging
import os
from collections import defaultdict
from concurrent.futures import as_completed, CancelledError, TimeoutError
from copy import deepcopy
from errno import EEXIST, ENOENT
from hashlib import md5
from io import StringIO
from os import environ, makedirs, stat, utime
from os.path import (
from posixpath import join as urljoin
from random import shuffle
from time import time
from threading import Thread
from queue import Queue
from queue import Empty as QueueEmpty
from urllib.parse import quote
import json
from swiftclient import Connection
from swiftclient.command_helpers import (
from swiftclient.utils import (
from swiftclient.exceptions import ClientException
from swiftclient.multithreading import MultiThreadingManager
def _download_container(self, container, options):
    _page_generator = self.list(container=container, options=options)
    try:
        next_page_downs = self._submit_page_downloads(container, _page_generator, options)
    except ClientException as err:
        if err.http_status != 404:
            raise
        raise SwiftError('Container %r not found' % container, container=container, exc=err)
    error = None
    while next_page_downs:
        page_downs = next_page_downs
        next_page_downs = None
        next_page_triggered = False
        next_page_trigger_point = 0.8 * len(page_downs)
        page_results_yielded = 0
        for o_down in interruptable_as_completed(page_downs):
            yield o_down.result()
            if not next_page_triggered:
                page_results_yielded += 1
                if page_results_yielded >= next_page_trigger_point:
                    try:
                        next_page_downs = self._submit_page_downloads(container, _page_generator, options)
                    except ClientException as err:
                        logger.exception(err)
                        error = err
                    except Exception:
                        for _d in page_downs:
                            _d.cancel()
                        raise
                    finally:
                        next_page_triggered = True
    if error:
        raise error