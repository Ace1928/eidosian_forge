import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LimitPreviousCommand(LimitCommand):
    """A command to limit the previous command."""
    commandmap = None

    def parsebit(self, pos):
        """Do nothing."""
        self.output = TaggedOutput().settag('span class="limits"')
        self.factory.clearskipped(pos)

    def __unicode__(self):
        """Return a printable representation."""
        return 'Limit previous command'