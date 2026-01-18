import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class FileSystemMovedEvent(FileSystemEvent):
    """
    File system event representing any kind of file system movement.
    """
    event_type = EVENT_TYPE_MOVED

    def __init__(self, src_path, dest_path):
        super(FileSystemMovedEvent, self).__init__(src_path)
        self._dest_path = dest_path

    @property
    def dest_path(self):
        """The destination path of the move event."""
        return self._dest_path

    @property
    def key(self):
        return (self.event_type, self.src_path, self.dest_path, self.is_directory)

    def __repr__(self):
        return '<%(class_name)s: src_path=%(src_path)r, dest_path=%(dest_path)r, is_directory=%(is_directory)s>' % dict(class_name=self.__class__.__name__, src_path=self.src_path, dest_path=self.dest_path, is_directory=self.is_directory)