import inspect
import select
import sys
import threading
import time
import traceback
import six
from ._abnf import ABNF
from ._core import WebSocket, getdefaulttimeout
from ._exceptions import *
from . import _logging

            Tears down the connection.
            If close_frame is set, we will invoke the on_close handler with the
            statusCode and reason from there.
            