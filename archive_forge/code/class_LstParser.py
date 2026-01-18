import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LstParser(object):
    """Parse global and local lstparams."""
    globalparams = dict()

    def parselstset(self, reader):
        """Parse a declaration of lstparams in lstset."""
        paramtext = self.extractlstset(reader)
        if not '{' in paramtext:
            Trace.error('Missing opening bracket in lstset: ' + paramtext)
            return
        lefttext = paramtext.split('{')[1]
        croppedtext = lefttext[:-1]
        LstParser.globalparams = self.parselstparams(croppedtext)

    def extractlstset(self, reader):
        """Extract the global lstset parameters."""
        paramtext = ''
        while not reader.finished():
            paramtext += reader.currentline()
            reader.nextline()
            if paramtext.endswith('}'):
                return paramtext
        Trace.error('Could not find end of \\lstset settings; aborting')

    def parsecontainer(self, container):
        """Parse some lstparams from elyxer.a container."""
        container.lstparams = LstParser.globalparams.copy()
        paramlist = container.getparameterlist('lstparams')
        container.lstparams.update(self.parselstparams(paramlist))

    def parselstparams(self, paramlist):
        """Process a number of lstparams from elyxer.a list."""
        paramdict = dict()
        for param in paramlist:
            if not '=' in param:
                if len(param.strip()) > 0:
                    Trace.error('Invalid listing parameter ' + param)
            else:
                key, value = param.split('=', 1)
                paramdict[key] = value
        return paramdict