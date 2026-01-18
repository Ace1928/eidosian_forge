import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MacroDefinition(CommandBit):
    """A function that defines a new command (a macro)."""
    macros = dict()

    def parsebit(self, pos):
        """Parse the function that defines the macro."""
        self.output = EmptyOutput()
        self.parameternumber = 0
        self.defaults = []
        self.factory.defining = True
        self.parseparameters(pos)
        self.factory.defining = False
        Trace.debug('New command ' + self.newcommand + ' (' + str(self.parameternumber) + ' parameters)')
        self.macros[self.newcommand] = self

    def parseparameters(self, pos):
        """Parse all optional parameters (number of parameters, default values)"""
        'and the mandatory definition.'
        self.newcommand = self.parsenewcommand(pos)
        literal = self.parsesquareliteral(pos)
        if literal:
            self.parameternumber = int(literal)
        bracket = self.parsesquare(pos)
        while bracket:
            self.defaults.append(bracket)
            bracket = self.parsesquare(pos)
        self.definition = self.parseparameter(pos)

    def parsenewcommand(self, pos):
        """Parse the name of the new command."""
        self.factory.clearskipped(pos)
        if self.factory.detecttype(Bracket, pos):
            return self.parseliteral(pos)
        if self.factory.detecttype(FormulaCommand, pos):
            return self.factory.create(FormulaCommand).extractcommand(pos)
        Trace.error('Unknown formula bit in defining function at ' + pos.identifier())
        return 'unknown'

    def instantiate(self):
        """Return an instance of the macro."""
        return self.definition.clone()