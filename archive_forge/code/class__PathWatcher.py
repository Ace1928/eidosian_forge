import threading
import sys
from os.path import basename
from _pydev_bundle import pydev_log
from os import scandir
import time
class _PathWatcher(object):
    """
    Helper to watch a single path.
    """

    def __init__(self, root_path, accept_directory, accept_file, single_visit_info, max_recursion_level, sleep_time=0.0):
        """
        :type root_path: str
        :type accept_directory: Callback[str, bool]
        :type accept_file: Callback[str, bool]
        :type max_recursion_level: int
        :type sleep_time: float
        """
        self.accept_directory = accept_directory
        self.accept_file = accept_file
        self._max_recursion_level = max_recursion_level
        self._root_path = root_path
        self.sleep_time = sleep_time
        self.sleep_at_elapsed = 1.0 / 30.0
        old_file_to_mtime = {}
        self._check(single_visit_info, lambda _change: None, old_file_to_mtime)

    def __eq__(self, o):
        if isinstance(o, _PathWatcher):
            return self._root_path == o._root_path
        return False

    def __ne__(self, o):
        return not self == o

    def __hash__(self):
        return hash(self._root_path)

    def _check_dir(self, dir_path, single_visit_info, append_change, old_file_to_mtime, level):
        if dir_path in single_visit_info.visited_dirs or level > self._max_recursion_level:
            return
        single_visit_info.visited_dirs.add(dir_path)
        try:
            if isinstance(dir_path, bytes):
                try:
                    dir_path = dir_path.decode(sys.getfilesystemencoding())
                except UnicodeDecodeError:
                    try:
                        dir_path = dir_path.decode('utf-8')
                    except UnicodeDecodeError:
                        return
            new_files = single_visit_info.file_to_mtime
            for entry in scandir(dir_path):
                single_visit_info.count += 1
                if single_visit_info.count % 300 == 0:
                    if self.sleep_time > 0:
                        t = time.time()
                        diff = t - single_visit_info.last_sleep_time
                        if diff > self.sleep_at_elapsed:
                            time.sleep(self.sleep_time)
                            single_visit_info.last_sleep_time = time.time()
                if entry.is_dir():
                    if self.accept_directory(entry.path):
                        self._check_dir(entry.path, single_visit_info, append_change, old_file_to_mtime, level + 1)
                elif self.accept_file(entry.path):
                    stat = entry.stat()
                    mtime = (stat.st_mtime_ns, stat.st_size)
                    path = entry.path
                    new_files[path] = mtime
                    old_mtime = old_file_to_mtime.pop(path, None)
                    if not old_mtime:
                        append_change((Change.added, path))
                    elif old_mtime != mtime:
                        append_change((Change.modified, path))
        except OSError:
            pass

    def _check(self, single_visit_info, append_change, old_file_to_mtime):
        self._check_dir(self._root_path, single_visit_info, append_change, old_file_to_mtime, 0)