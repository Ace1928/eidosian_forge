import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization
def _update_data_from_dir(self, path):
    data = {'iterations': [], 'meta': {}}
    training_json = os.path.join(path, 'catboost_training.json')
    if os.path.isfile(training_json):
        try:
            with open(training_json, 'r') as json_data:
                training_data = json.load(json_data)
                data['meta'] = training_data['meta']
                data['iterations'] = training_data['iterations']
        except ValueError:
            pass
    return {'passed_iterations': data['iterations'][-1]['iteration'] if data['iterations'] else 0, 'total_iterations': data['meta']['iteration_count'] if data['meta'] else 0, 'data': data}