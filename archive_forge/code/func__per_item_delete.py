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
def _per_item_delete(self, container, objects, options, rdict, rq):
    for delete_obj in objects:
        obj = delete_obj.object_name
        obj_options = dict(options, **delete_obj.options or {})
        obj_del = self.thread_manager.object_dd_pool.submit(self._delete_object, container, obj, obj_options, results_queue=rq)
        obj_details = {'container': container, 'object': obj}
        rdict[obj_del] = obj_details