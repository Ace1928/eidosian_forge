import xml.dom.minidom, xml.sax.saxutils
import os, time, fcntl
from xdg.Exceptions import ParsingError
class RecentFile:

    def __init__(self):
        self.URI = ''
        self.MimeType = ''
        self.Timestamp = ''
        self.Private = False
        self.Groups = []

    def __cmp__(self, other):
        return cmp(self.Timestamp, other.Timestamp)

    def __lt__(self, other):
        return self.Timestamp < other.Timestamp

    def __eq__(self, other):
        return self.URI == str(other)

    def __str__(self):
        return self.URI