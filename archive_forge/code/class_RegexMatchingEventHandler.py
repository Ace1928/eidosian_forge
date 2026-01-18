import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class RegexMatchingEventHandler(FileSystemEventHandler):
    """
    Matches given regexes with file paths associated with occurring events.
    """

    def __init__(self, regexes=['.*'], ignore_regexes=[], ignore_directories=False, case_sensitive=False):
        super(RegexMatchingEventHandler, self).__init__()
        if case_sensitive:
            self._regexes = [re.compile(r) for r in regexes]
            self._ignore_regexes = [re.compile(r) for r in ignore_regexes]
        else:
            self._regexes = [re.compile(r, re.I) for r in regexes]
            self._ignore_regexes = [re.compile(r, re.I) for r in ignore_regexes]
        self._ignore_directories = ignore_directories
        self._case_sensitive = case_sensitive

    @property
    def regexes(self):
        """
        (Read-only)
        Regexes to allow matching event paths.
        """
        return self._regexes

    @property
    def ignore_regexes(self):
        """
        (Read-only)
        Regexes to ignore matching event paths.
        """
        return self._ignore_regexes

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
        if any((r.match(p) for r in self.ignore_regexes for p in paths)):
            return
        if any((r.match(p) for r in self.regexes for p in paths)):
            self.on_any_event(event)
            _method_map = {EVENT_TYPE_MODIFIED: self.on_modified, EVENT_TYPE_MOVED: self.on_moved, EVENT_TYPE_CREATED: self.on_created, EVENT_TYPE_DELETED: self.on_deleted}
            event_type = event.event_type
            _method_map[event_type](event)