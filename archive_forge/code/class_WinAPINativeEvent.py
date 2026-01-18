from __future__ import with_statement
import ctypes.wintypes
from functools import reduce
class WinAPINativeEvent(object):

    def __init__(self, action, src_path):
        self.action = action
        self.src_path = src_path

    @property
    def is_added(self):
        return self.action == FILE_ACTION_CREATED

    @property
    def is_removed(self):
        return self.action == FILE_ACTION_REMOVED

    @property
    def is_modified(self):
        return self.action == FILE_ACTION_MODIFIED

    @property
    def is_renamed_old(self):
        return self.action == FILE_ACTION_RENAMED_OLD_NAME

    @property
    def is_renamed_new(self):
        return self.action == FILE_ACTION_RENAMED_NEW_NAME

    def __repr__(self):
        return '<WinAPINativeEvent: action=%d, src_path=%r>' % (self.action, self.src_path)