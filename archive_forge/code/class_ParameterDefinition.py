import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class ParameterDefinition(object):
    """The definition of a parameter in a hybrid function."""
    '[] parameters are optional, {} parameters are mandatory.'
    'Each parameter has a one-character name, like {$1} or {$p}.'
    'A parameter that ends in ! like {$p!} is a literal.'
    'Example: [$1]{$p!} reads an optional parameter $1 and a literal mandatory parameter p.'
    parambrackets = [('[', ']'), ('{', '}')]

    def __init__(self):
        self.name = None
        self.literal = False
        self.optional = False
        self.value = None
        self.literalvalue = None

    def parse(self, pos):
        """Parse a parameter definition: [$0], {$x}, {$1!}..."""
        for opening, closing in ParameterDefinition.parambrackets:
            if pos.checkskip(opening):
                if opening == '[':
                    self.optional = True
                if not pos.checkskip('$'):
                    Trace.error('Wrong parameter name, did you mean $' + pos.current() + '?')
                    return None
                self.name = pos.skipcurrent()
                if pos.checkskip('!'):
                    self.literal = True
                if not pos.checkskip(closing):
                    Trace.error('Wrong parameter closing ' + pos.skipcurrent())
                    return None
                return self
        Trace.error('Wrong character in parameter template: ' + pos.skipcurrent())
        return None

    def read(self, pos, function):
        """Read the parameter itself using the definition."""
        if self.literal:
            if self.optional:
                self.literalvalue = function.parsesquareliteral(pos)
            else:
                self.literalvalue = function.parseliteral(pos)
            if self.literalvalue:
                self.value = FormulaConstant(self.literalvalue)
        elif self.optional:
            self.value = function.parsesquare(pos)
        else:
            self.value = function.parseparameter(pos)

    def __unicode__(self):
        """Return a printable representation."""
        result = 'param ' + self.name
        if self.value:
            result += ': ' + str(self.value)
        else:
            result += ' (empty)'
        return result