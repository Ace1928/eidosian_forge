from __future__ import unicode_literals
import os.path  # splitext
import pkg_resources
from pybtex.exceptions import PybtexError
class PluginGroupNotFound(PybtexError):

    def __init__(self, group_name):
        message = u'plugin group {group_name} not found'.format(group_name=group_name)
        super(PluginGroupNotFound, self).__init__(message)