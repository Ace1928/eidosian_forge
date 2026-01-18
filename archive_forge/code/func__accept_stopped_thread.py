import os
import re
import sys
import typing as t
from pathlib import Path
import zmq
from IPython.core.getipython import get_ipython
from IPython.core.inputtransformer2 import leading_empty_lines
from tornado.locks import Event
from tornado.queues import Queue
from zmq.utils import jsonapi
from .compiler import get_file_name, get_tmp_directory, get_tmp_hash_seed
def _accept_stopped_thread(self, thread_name):
    forbid_list = ['IPythonHistorySavingThread', 'Thread-2', 'Thread-3', 'Thread-4']
    return thread_name not in forbid_list