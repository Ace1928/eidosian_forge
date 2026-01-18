import sys
from functools import wraps
from .domwidget import DOMWidget
from .trait_types import TypedTuple
from .widget import register
from .._version import __jupyter_widgets_output_version__
from traitlets import Unicode, Dict
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import clear_output
from IPython import get_ipython
import traceback
def _append_stream_output(self, text, stream_name):
    """Append a stream output."""
    self.outputs += ({'output_type': 'stream', 'name': stream_name, 'text': text},)