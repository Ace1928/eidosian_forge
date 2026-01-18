import os
from dulwich.errors import (
from dulwich.objects import (
from dulwich.file import (
class _InotifyRefsWatcher(object):

    def __init__(self, path):
        import pyinotify
        from queue import Queue
        self.path = os.fsdecode(path)
        self.manager = pyinotify.WatchManager()
        self.manager.add_watch(self.path, pyinotify.IN_DELETE | pyinotify.IN_CLOSE_WRITE | pyinotify.IN_MOVED_TO, rec=True, auto_add=True)
        self.notifier = pyinotify.ThreadedNotifier(self.manager, default_proc_fun=self._notify)
        self.queue = Queue()

    def _notify(self, event):
        if event.dir:
            return
        if event.pathname.endswith('.lock'):
            return
        ref = os.fsencode(os.path.relpath(event.pathname, self.path))
        if event.maskname == 'IN_DELETE':
            self.queue.put_nowait((ref, None))
        elif event.maskname in ('IN_CLOSE_WRITE', 'IN_MOVED_TO'):
            with open(event.pathname, 'rb') as f:
                sha = f.readline().rstrip(b'\n\r')
                self.queue.put_nowait((ref, sha))

    def __next__(self):
        return self.queue.get()

    def __enter__(self):
        self.notifier.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.notifier.stop()
        return False