from __future__ import unicode_literals
import typing
import threading
from six.moves.queue import Queue
from .copy import copy_file_internal, copy_modified_time
from .errors import BulkCopyFailed
from .tools import copy_file_data
class Copier(object):
    """Copy files in worker threads."""

    def __init__(self, num_workers=4, preserve_time=False):
        if num_workers < 0:
            raise ValueError('num_workers must be >= 0')
        self.num_workers = num_workers
        self.preserve_time = preserve_time
        self.all_tasks = []
        self.queue = None
        self.workers = []
        self.errors = []
        self.running = False

    def start(self):
        """Start the workers."""
        if self.num_workers:
            self.queue = Queue(maxsize=self.num_workers)
            self.workers = [_Worker(self) for _ in range(self.num_workers)]
            for worker in self.workers:
                worker.start()
        self.running = True

    def stop(self):
        """Stop the workers (will block until they are finished)."""
        if self.running and self.num_workers:
            for _worker in self.workers:
                self.queue.put(None)
            for worker in self.workers:
                worker.join()
            if self.preserve_time:
                for args in self.all_tasks:
                    copy_modified_time(*args)
            del self.workers[:]
            self.queue.join()
        self.running = False

    def add_error(self, error):
        """Add an exception raised by a task."""
        self.errors.append(error)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        if traceback is None and self.errors:
            raise BulkCopyFailed(self.errors)

    def copy(self, src_fs, src_path, dst_fs, dst_path, preserve_time=False):
        """Copy a file from one fs to another."""
        if self.queue is None:
            copy_file_internal(src_fs, src_path, dst_fs, dst_path, preserve_time=self.preserve_time)
        else:
            self.all_tasks.append((src_fs, src_path, dst_fs, dst_path))
            src_file = src_fs.openbin(src_path, 'r')
            try:
                dst_file = dst_fs.openbin(dst_path, 'w')
            except Exception:
                src_file.close()
                raise
            task = _CopyTask(src_file, dst_file)
            self.queue.put(task)