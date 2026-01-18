import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class HybridFunction(ParameterFunction):
    """
  A parameter function where the output is also defined using a template.
  The template can use a number of functions; each function has an associated
  tag.
  Example: [f0{$1},span class="fbox"] defines a function f0 which corresponds
  to a span of class fbox, yielding <span class="fbox">$1</span>.
  Literal parameters can be used in tags definitions:
    [f0{$1},span style="color: $p;"]
  yields <span style="color: $p;">$1</span>, where $p is a literal parameter.
  Sizes can be specified in hybridsizes, e.g. adding parameter sizes. By
  default the resulting size is the max of all arguments. Sizes are used
  to generate the right parameters.
  A function followed by a single / is output as a self-closing XHTML tag:
    [f0/,hr]
  will generate <hr/>.
  """
    commandmap = FormulaConfig.hybridfunctions

    def parsebit(self, pos):
        """Parse a function with [] and {} parameters"""
        readtemplate = self.translated[0]
        writetemplate = self.translated[1]
        self.readparams(readtemplate, pos)
        self.contents = self.writeparams(writetemplate)
        self.computehybridsize()

    def writeparams(self, writetemplate):
        """Write all params according to the template"""
        return self.writepos(TextPosition(writetemplate))

    def writepos(self, pos):
        """Write all params as read in the parse position."""
        result = []
        while not pos.finished():
            if pos.checkskip('$'):
                param = self.writeparam(pos)
                if param:
                    result.append(param)
            elif pos.checkskip('f'):
                function = self.writefunction(pos)
                if function:
                    function.type = None
                    result.append(function)
            elif pos.checkskip('('):
                result.append(self.writebracket('left', '('))
            elif pos.checkskip(')'):
                result.append(self.writebracket('right', ')'))
            else:
                result.append(FormulaConstant(pos.skipcurrent()))
        return result

    def writeparam(self, pos):
        """Write a single param of the form $0, $x..."""
        name = '$' + pos.skipcurrent()
        if not name in self.params:
            Trace.error('Unknown parameter ' + name)
            return None
        if not self.params[name]:
            return None
        if pos.checkskip('.'):
            self.params[name].value.type = pos.globalpha()
        return self.params[name].value

    def writefunction(self, pos):
        """Write a single function f0,...,fn."""
        tag = self.readtag(pos)
        if not tag:
            return None
        if pos.checkskip('/'):
            return TaggedBit().selfcomplete(tag)
        if not pos.checkskip('{'):
            Trace.error('Function should be defined in {}')
            return None
        pos.pushending('}')
        contents = self.writepos(pos)
        pos.popending()
        if len(contents) == 0:
            return None
        return TaggedBit().complete(contents, tag)

    def readtag(self, pos):
        """Get the tag corresponding to the given index. Does parameter substitution."""
        if not pos.current().isdigit():
            Trace.error('Function should be f0,...,f9: f' + pos.current())
            return None
        index = int(pos.skipcurrent())
        if 2 + index > len(self.translated):
            Trace.error('Function f' + str(index) + ' is not defined')
            return None
        tag = self.translated[2 + index]
        if not '$' in tag:
            return tag
        for variable in self.params:
            if variable in tag:
                param = self.params[variable]
                if not param.literal:
                    Trace.error('Parameters in tag ' + tag + ' should be literal: {' + variable + '!}')
                    continue
                if param.literalvalue:
                    value = param.literalvalue
                else:
                    value = ''
                tag = tag.replace(variable, value)
        return tag

    def writebracket(self, direction, character):
        """Return a new bracket looking at the given direction."""
        return self.factory.create(BracketCommand).create(direction, character)

    def computehybridsize(self):
        """Compute the size of the hybrid function."""
        if not self.command in HybridSize.configsizes:
            self.computesize()
            return
        self.size = HybridSize().getsize(self)
        for element in self.contents:
            element.size = self.size