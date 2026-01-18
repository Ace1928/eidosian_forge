import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
def _run_update(self):
    self.thread = Thread(target=self.start, args=())
    self.thread.start()