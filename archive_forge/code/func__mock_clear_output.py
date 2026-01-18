import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def _mock_clear_output(self):
    """ Mock function that records calls to it """
    calls = []

    def clear_output(*args, **kwargs):
        calls.append((args, kwargs))
    clear_output.calls = calls
    return clear_output