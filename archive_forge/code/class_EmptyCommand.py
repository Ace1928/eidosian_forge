import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class EmptyCommand(CommandBit):
    """An empty command (without parameters)"""
    commandmap = FormulaConfig.commands

    def parsebit(self, pos):
        """Parse a command without parameters"""
        self.contents = [FormulaConstant(self.translated)]