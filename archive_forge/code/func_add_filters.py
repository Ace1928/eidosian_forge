import errno
import io
import logging
import logging.handlers
import os
import queue
import re
import struct
import threading
import traceback
from socketserver import ThreadingTCPServer, StreamRequestHandler
def add_filters(self, filterer, filters):
    """Add filters to a filterer from a list of names."""
    for f in filters:
        try:
            if callable(f) or callable(getattr(f, 'filter', None)):
                filter_ = f
            else:
                filter_ = self.config['filters'][f]
            filterer.addFilter(filter_)
        except Exception as e:
            raise ValueError('Unable to add filter %r' % f) from e