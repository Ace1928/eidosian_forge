import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class DirDeletedEvent(FileSystemEvent):
    """File system event representing directory deletion on the file system."""
    event_type = EVENT_TYPE_DELETED
    is_directory = True

    def __init__(self, src_path):
        super(DirDeletedEvent, self).__init__(src_path)

    def __repr__(self):
        return '<%(class_name)s: src_path=%(src_path)r>' % dict(class_name=self.__class__.__name__, src_path=self.src_path)