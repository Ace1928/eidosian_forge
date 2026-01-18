import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MacroFunction(CommandBit):
    """A function that was defined using a macro."""
    commandmap = MacroDefinition.macros

    def parsebit(self, pos):
        """Parse a number of input parameters."""
        self.output = FilteredOutput()
        self.values = []
        macro = self.translated
        self.parseparameters(pos, macro)
        self.completemacro(macro)

    def parseparameters(self, pos, macro):
        """Parse as many parameters as are needed."""
        self.parseoptional(pos, list(macro.defaults))
        self.parsemandatory(pos, macro.parameternumber - len(macro.defaults))
        if len(self.values) < macro.parameternumber:
            Trace.error('Missing parameters in macro ' + str(self))

    def parseoptional(self, pos, defaults):
        """Parse optional parameters."""
        optional = []
        while self.factory.detecttype(SquareBracket, pos):
            optional.append(self.parsesquare(pos))
            if len(optional) > len(defaults):
                break
        for value in optional:
            default = defaults.pop()
            if len(value.contents) > 0:
                self.values.append(value)
            else:
                self.values.append(default)
        self.values += defaults

    def parsemandatory(self, pos, number):
        """Parse a number of mandatory parameters."""
        for index in range(number):
            parameter = self.parsemacroparameter(pos, number - index)
            if not parameter:
                return
            self.values.append(parameter)

    def parsemacroparameter(self, pos, remaining):
        """Parse a macro parameter. Could be a bracket or a single letter."""
        'If there are just two values remaining and there is a running number,'
        'parse as two separater numbers.'
        self.factory.clearskipped(pos)
        if pos.finished():
            return None
        if self.factory.detecttype(FormulaNumber, pos):
            return self.parsenumbers(pos, remaining)
        return self.parseparameter(pos)

    def parsenumbers(self, pos, remaining):
        """Parse the remaining parameters as a running number."""
        'For example, 12 would be {1}{2}.'
        number = self.factory.parsetype(FormulaNumber, pos)
        if not len(number.original) == remaining:
            return number
        for digit in number.original:
            value = self.factory.create(FormulaNumber)
            value.add(FormulaConstant(digit))
            value.type = number
            self.values.append(value)
        return None

    def completemacro(self, macro):
        """Complete the macro with the parameters read."""
        self.contents = [macro.instantiate()]
        replaced = [False] * len(self.values)
        for parameter in self.searchall(MacroParameter):
            index = parameter.number - 1
            if index >= len(self.values):
                Trace.error('Macro parameter index out of bounds: ' + str(index))
                return
            replaced[index] = True
            parameter.contents = [self.values[index].clone()]
        for index in range(len(self.values)):
            if not replaced[index]:
                self.addfilter(index, self.values[index])

    def addfilter(self, index, value):
        """Add a filter for the given parameter number and parameter value."""
        original = '#' + str(index + 1)
        value = ''.join(self.values[0].gethtml())
        self.output.addfilter(original, value)