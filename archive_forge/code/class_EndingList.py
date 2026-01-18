import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class EndingList(object):
    """A list of position endings"""

    def __init__(self):
        self.endings = []

    def add(self, ending, optional=False):
        """Add a new ending to the list"""
        self.endings.append(PositionEnding(ending, optional))

    def pickpending(self, pos):
        """Pick any pending endings from a parse position."""
        self.endings += pos.endinglist.endings

    def checkin(self, pos):
        """Search for an ending"""
        if self.findending(pos):
            return True
        return False

    def pop(self, pos):
        """Remove the ending at the current position"""
        if pos.isout():
            Trace.error('No ending out of bounds')
            return ''
        ending = self.findending(pos)
        if not ending:
            Trace.error('No ending at ' + pos.current())
            return ''
        for each in reversed(self.endings):
            self.endings.remove(each)
            if each == ending:
                return each.ending
            elif not each.optional:
                Trace.error('Removed non-optional ending ' + each)
        Trace.error('No endings left')
        return ''

    def findending(self, pos):
        """Find the ending at the current position"""
        if len(self.endings) == 0:
            return None
        for index, ending in enumerate(reversed(self.endings)):
            if ending.checkin(pos):
                return ending
            if not ending.optional:
                return None
        return None

    def checkpending(self):
        """Check if there are any pending endings"""
        if len(self.endings) != 0:
            Trace.error('Pending ' + str(self) + ' left open')

    def __unicode__(self):
        """Printable representation"""
        string = 'endings ['
        for ending in self.endings:
            string += str(ending) + ','
        if len(self.endings) > 0:
            string = string[:-1]
        return string + ']'