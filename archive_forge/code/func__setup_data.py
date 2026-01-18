from parlai.core.teachers import FixedDialogTeacher
from .build import build
import json
import os
def _setup_data(self, path):
    print('loading: ', path)
    with open(path) as data_file:
        self.examples = json.load(data_file)