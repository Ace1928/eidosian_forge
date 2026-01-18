import csv
import json
import logging
class _ReaderWrapper(object):
    """A wrapper for csv.reader / csv.DictReader to make it picklable."""

    def __init__(self, line_generator, column_names, delimiter, decode_to_dict, skip_initial_space):
        self._state = (line_generator, column_names, delimiter, decode_to_dict, skip_initial_space)
        self._line_generator = line_generator
        if decode_to_dict:
            self._reader = csv.DictReader(line_generator, column_names, delimiter=str(delimiter), skipinitialspace=skip_initial_space)
        else:
            self._reader = csv.reader(line_generator, delimiter=str(delimiter), skipinitialspace=skip_initial_space)

    def read_record(self, x):
        self._line_generator.push_line(x)
        return self._reader.next()

    def __getstate__(self):
        return self._state

    def __setstate__(self, state):
        self.__init__(*state)