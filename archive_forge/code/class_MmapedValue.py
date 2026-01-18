import os
from threading import Lock
import warnings
from .mmap_dict import mmap_key, MmapedDict
class MmapedValue:
    """A float protected by a mutex backed by a per-process mmaped file."""
    _multiprocess = True

    def __init__(self, typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode='', **kwargs):
        self._params = (typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode)
        if 'prometheus_multiproc_dir' in os.environ and 'PROMETHEUS_MULTIPROC_DIR' not in os.environ:
            os.environ['PROMETHEUS_MULTIPROC_DIR'] = os.environ['prometheus_multiproc_dir']
            warnings.warn('prometheus_multiproc_dir variable has been deprecated in favor of the upper case naming PROMETHEUS_MULTIPROC_DIR', DeprecationWarning)
        with lock:
            self.__check_for_pid_change()
            self.__reset()
            values.append(self)

    def __reset(self):
        typ, metric_name, name, labelnames, labelvalues, help_text, multiprocess_mode = self._params
        if typ == 'gauge':
            file_prefix = typ + '_' + multiprocess_mode
        else:
            file_prefix = typ
        if file_prefix not in files:
            filename = os.path.join(os.environ.get('PROMETHEUS_MULTIPROC_DIR'), '{}_{}.db'.format(file_prefix, pid['value']))
            files[file_prefix] = MmapedDict(filename)
        self._file = files[file_prefix]
        self._key = mmap_key(metric_name, name, labelnames, labelvalues, help_text)
        self._value, self._timestamp = self._file.read_value(self._key)

    def __check_for_pid_change(self):
        actual_pid = process_identifier()
        if pid['value'] != actual_pid:
            pid['value'] = actual_pid
            for f in files.values():
                f.close()
            files.clear()
            for value in values:
                value.__reset()

    def inc(self, amount):
        with lock:
            self.__check_for_pid_change()
            self._value += amount
            self._timestamp = 0.0
            self._file.write_value(self._key, self._value, self._timestamp)

    def set(self, value, timestamp=None):
        with lock:
            self.__check_for_pid_change()
            self._value = value
            self._timestamp = timestamp or 0.0
            self._file.write_value(self._key, self._value, self._timestamp)

    def set_exemplar(self, exemplar):
        return

    def get(self):
        with lock:
            self.__check_for_pid_change()
            return self._value

    def get_exemplar(self):
        return None