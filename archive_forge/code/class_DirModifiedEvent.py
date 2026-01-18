import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class DirModifiedEvent(FileSystemEvent):
    """
    File system event representing directory modification on the file system.
    """
    event_type = EVENT_TYPE_MODIFIED
    is_directory = True

    def __init__(self, src_path):
        super(DirModifiedEvent, self).__init__(src_path)

    def __repr__(self):
        return '<%(class_name)s: src_path=%(src_path)r>' % dict(class_name=self.__class__.__name__, src_path=self.src_path)