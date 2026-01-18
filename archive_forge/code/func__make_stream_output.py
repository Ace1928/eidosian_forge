import sys
from unittest import TestCase
from contextlib import contextmanager
from IPython.display import Markdown, Image
from ipywidgets import widget_output
def _make_stream_output(text, name):
    return {'output_type': 'stream', 'name': name, 'text': text}