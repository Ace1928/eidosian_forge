import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
def _update_data(self):
    data = {}
    dirs = [{'name': name, 'path': path} for name, path in zip(self._names, self._train_dirs)]
    all_completed = True
    for dir_info in dirs:
        path = dir_info.get('path')
        content = self._update_data_from_dir(path)
        if not content:
            continue
        data[path] = {'path': path, 'name': dir_info.get('name'), 'content': content}
        passed_iterations = data[path]['content']['passed_iterations']
        total_iterations = data[path]['content']['total_iterations']
        all_completed &= passed_iterations + 1 >= total_iterations and total_iterations != 0
    if all_completed:
        self._need_to_stop.set()
    self.data = data