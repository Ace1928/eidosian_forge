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
class SingleFigurePage(tornado.web.RequestHandler):

    def __init__(self, application, request, *, url_prefix='', **kwargs):
        self.url_prefix = url_prefix
        super().__init__(application, request, **kwargs)

    def get(self, fignum):
        fignum = int(fignum)
        manager = Gcf.get_fig_manager(fignum)
        ws_uri = f'ws://{self.request.host}{self.url_prefix}/'
        self.render('single_figure.html', prefix=self.url_prefix, ws_uri=ws_uri, fig_id=fignum, toolitems=core.NavigationToolbar2WebAgg.toolitems, canvas=manager.canvas)