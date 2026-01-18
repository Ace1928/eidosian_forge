import inspect
import os
import threading
from contextlib import contextmanager
from datetime import datetime
from traits import trait_notifiers
class MultiThreadRecordContainer(object):
    """ A container of record containers that are used by separate threads.

    Each record container is mapped to a thread name id. When a RecordContainer
    does not exist for a specific thread a new empty RecordContainer will be
    created on request.


    """

    def __init__(self):
        self._creation_lock = threading.Lock()
        self._record_containers = {}

    def get_change_event_collector(self, thread_name):
        """ Return the dedicated RecordContainer for the thread.

        If no RecordContainer is found for `thread_name` then a new
        RecordContainer is created.

        """
        with self._creation_lock:
            container = self._record_containers.get(thread_name)
            if container is None:
                container = RecordContainer()
                self._record_containers[thread_name] = container
            return container

    def save_to_directory(self, directory_name):
        """ Save records files into the directory.

        Each RecordContainer will dump its records on a separate file named
        <thread_name>.trace.

        """
        with self._creation_lock:
            containers = self._record_containers
            for thread_name, container in containers.items():
                filename = os.path.join(directory_name, '{0}.trace'.format(thread_name))
                container.save_to_file(filename)