import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class RecordContainer(object):
    """ A simple record container.

     This class is commonly used to hold records from a single thread.

    """

    def __init__(self):
        self._records = []

    def record(self, record):
        """ Add the record into the container.

        """
        self._records.append(record)

    def save_to_file(self, filename):
        """ Save the records into a file.

        """
        with open(filename, 'w', encoding='utf-8') as fh:
            for record in self._records:
                fh.write(str(record))