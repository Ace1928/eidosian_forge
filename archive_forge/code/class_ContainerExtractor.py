import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ContainerExtractor(object):
    """A class to extract certain containers."""

    def __init__(self, config):
        """The config parameter is a map containing three lists: allowed, copied and extracted."""
        'Each of the three is a list of class names for containers.'
        'Allowed containers are included as is into the result.'
        'Cloned containers are cloned and placed into the result.'
        'Extracted containers are looked into.'
        'All other containers are silently ignored.'
        self.allowed = config['allowed']
        self.cloned = config['cloned']
        self.extracted = config['extracted']

    def extract(self, container):
        """Extract a group of selected containers from elyxer.a container."""
        list = []
        locate = lambda c: c.__class__.__name__ in self.allowed + self.cloned
        recursive = lambda c: c.__class__.__name__ in self.extracted
        process = lambda c: self.process(c, list)
        container.recursivesearch(locate, recursive, process)
        return list

    def process(self, container, list):
        """Add allowed containers, clone cloned containers and add the clone."""
        name = container.__class__.__name__
        if name in self.allowed:
            list.append(container)
        elif name in self.cloned:
            list.append(self.safeclone(container))
        else:
            Trace.error('Unknown container class ' + name)

    def safeclone(self, container):
        """Return a new container with contents only in a safe list, recursively."""
        clone = Cloner.clone(container)
        clone.output = container.output
        clone.contents = self.extract(container)
        return clone