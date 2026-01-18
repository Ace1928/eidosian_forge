from __future__ import (absolute_import, division, print_function)
import csv
import datetime
import os
import time
import threading
from abc import ABCMeta, abstractmethod
from functools import partial
from ansible.module_utils._text import to_bytes, to_text
from ansible.module_utils.six import with_metaclass
from ansible.parsing.ajson import AnsibleJSONEncoder, json
from ansible.plugins.callback import CallbackBase
def _open_files(self, task_uuid=None):
    output_format = self._output_format
    output_dir = self._output_dir
    for feature in self._features:
        data = {b'counter': to_bytes(self._counter), b'task_uuid': to_bytes(task_uuid), b'feature': to_bytes(feature), b'ext': to_bytes(output_format)}
        if self._files.get(feature):
            try:
                self._files[feature].close()
            except Exception:
                pass
        if self.write_files:
            filename = self._file_name_format % data
            self._files[feature] = open(os.path.join(output_dir, filename), 'w+')
            if output_format == b'csv':
                self._writers[feature] = partial(csv_writer, csv.writer(self._files[feature]))
            elif output_format == b'json':
                self._writers[feature] = partial(json_writer, self._files[feature])