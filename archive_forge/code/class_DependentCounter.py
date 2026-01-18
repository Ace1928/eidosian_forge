import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class DependentCounter(NumberCounter):
    """A counter which depends on another one (the master)."""

    def setmaster(self, master):
        """Set the master counter."""
        self.master = master
        self.last = self.master.getvalue()
        return self

    def getnext(self):
        """Increase or, if the master counter has changed, restart."""
        if self.last != self.master.getvalue():
            self.reset()
        value = NumberCounter.getnext(self)
        self.last = self.master.getvalue()
        return value

    def getvalue(self):
        """Get the value of the combined counter: master.dependent."""
        return self.master.getvalue() + '.' + NumberCounter.getvalue(self)