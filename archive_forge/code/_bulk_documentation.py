from __future__ import unicode_literals
import typing
import threading
from six.moves.queue import Queue
from .copy import copy_file_internal, copy_modified_time
from .errors import BulkCopyFailed
from .tools import copy_file_data
Copy a file from one fs to another.