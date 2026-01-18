import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
def _stop_update(self):
    self._update_after_stop_signal = True
    self._need_to_stop.set()
    self.thread.join()