from contextlib import contextmanager
import errno
from io import BytesIO
import json
import mimetypes
from pathlib import Path
import random
import sys
import signal
import threading
import tornado.web
import tornado.ioloop
import tornado.websocket
import matplotlib as mpl
from matplotlib.backend_bases import _Backend
from matplotlib._pylab_helpers import Gcf
from . import backend_webagg_core as core
from .backend_webagg_core import (  # noqa: F401 # pylint: disable=W0611

            Generate a list of n random ports near the given port.

            The first 5 ports will be sequential, and the remaining n-5 will be
            randomly selected in the range [port-2*n, port+2*n].
            