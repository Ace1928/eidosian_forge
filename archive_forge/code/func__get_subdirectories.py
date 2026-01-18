import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
def _get_subdirectories(self, a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]