import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def _mock_get_ipython(self, msg_id):
    """ Returns a mock IPython application with a mocked kernel """
    kernel = type('mock_kernel', (object,), {'_parent_header': {'header': {'msg_id': msg_id}}})

    def showtraceback(self_, exc_tuple, *args, **kwargs):
        etype, evalue, tb = exc_tuple
        raise etype(evalue)
    ipython = type('mock_ipython', (object,), {'kernel': kernel, 'showtraceback': showtraceback})
    return ipython