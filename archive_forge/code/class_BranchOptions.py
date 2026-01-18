import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BranchOptions(object):
    """A set of options for a branch"""

    def __init__(self, name):
        self.name = name
        self.options = {'color': '#ffffff'}

    def set(self, key, value):
        """Set a branch option"""
        if not key.startswith(ContainerConfig.string['startcommand']):
            Trace.error('Invalid branch option ' + key)
            return
        key = key.replace(ContainerConfig.string['startcommand'], '')
        self.options[key] = value

    def isselected(self):
        """Return if the branch is selected"""
        if not 'selected' in self.options:
            return False
        return self.options['selected'] == '1'

    def __unicode__(self):
        """String representation"""
        return 'options for ' + self.name + ': ' + str(self.options)