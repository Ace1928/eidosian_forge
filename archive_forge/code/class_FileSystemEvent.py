import os.path
import logging
import re
from .patterns import match_any_paths
from wandb_watchdog.utils import has_attribute
from wandb_watchdog.utils import unicode_paths
class FileSystemEvent(object):
    """
    Immutable type that represents a file system event that is triggered
    when a change occurs on the monitored file system.

    All FileSystemEvent objects are required to be immutable and hence
    can be used as keys in dictionaries or be added to sets.
    """
    event_type = None
    'The type of the event as a string.'
    is_directory = False
    'True if event was emitted for a directory; False otherwise.'

    def __init__(self, src_path):
        self._src_path = src_path

    @property
    def src_path(self):
        """Source path of the file system object that triggered this event."""
        return self._src_path

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%(class_name)s: event_type=%(event_type)s, src_path=%(src_path)r, is_directory=%(is_directory)s>' % dict(class_name=self.__class__.__name__, event_type=self.event_type, src_path=self.src_path, is_directory=self.is_directory)

    @property
    def key(self):
        return (self.event_type, self.src_path, self.is_directory)

    def __eq__(self, event):
        return self.key == event.key

    def __ne__(self, event):
        return self.key != event.key

    def __hash__(self):
        return hash(self.key)