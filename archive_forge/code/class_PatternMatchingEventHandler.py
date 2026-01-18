import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class PatternMatchingEventHandler(FileSystemEventHandler):
    """
    Matches given patterns with file paths associated with occurring events.
    """

    def __init__(self, patterns=None, ignore_patterns=None, ignore_directories=False, case_sensitive=False):
        super(PatternMatchingEventHandler, self).__init__()
        self._patterns = patterns
        self._ignore_patterns = ignore_patterns
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def patterns(self):
        """
        (Read-only)
        Patterns to allow matching event paths.
        """
        return self._patterns

    @property
    def ignore_patterns(self):
        """
        (Read-only)
        Patterns to ignore matching event paths.
        """
        return self._ignore_patterns

    @property
    def ignore_directories(self):
        """
        (Read-only)
        ``True`` if directories should be ignored; ``False`` otherwise.
        """
        return self._ignore_directories

    @property
    def case_sensitive(self):
        """
        (Read-only)
        ``True`` if path names should be matched sensitive to case; ``False``
        otherwise.
        """
        return self._case_sensitive

    def dispatch(self, event):
        """Dispatches events to the appropriate methods.

        :param event:
            The event object representing the file system event.
        :type event:
            :class:`FileSystemEvent`
        """
        if self.ignore_directories and event.is_directory:
            return
        paths = []
        if has_attribute(event, 'dest_path'):
            paths.append(unicode_paths.decode(event.dest_path))
        if event.src_path:
            paths.append(unicode_paths.decode(event.src_path))
        if match_any_paths(paths, included_patterns=self.patterns, excluded_patterns=self.ignore_patterns, case_sensitive=self.case_sensitive):
            self.on_any_event(event)
            _method_map = {EVENT_TYPE_MODIFIED: self.on_modified, EVENT_TYPE_MOVED: self.on_moved, EVENT_TYPE_CREATED: self.on_created, EVENT_TYPE_DELETED: self.on_deleted}
            event_type = event.event_type
            _method_map[event_type](event)