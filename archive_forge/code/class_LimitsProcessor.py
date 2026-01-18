import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LimitsProcessor(MathsProcessor):
    """A processor for limits inside an element."""

    def process(self, contents, index):
        """Process the limits for an element."""
        if Options.simplemath:
            return
        if self.checklimits(contents, index):
            self.modifylimits(contents, index)
        if self.checkscript(contents, index) and self.checkscript(contents, index + 1):
            self.modifyscripts(contents, index)

    def checklimits(self, contents, index):
        """Check if the current position has a limits command."""
        if not DocumentParameters.displaymode:
            return False
        if self.checkcommand(contents, index + 1, LimitPreviousCommand):
            self.limitsahead(contents, index)
            return False
        if not isinstance(contents[index], LimitCommand):
            return False
        return self.checkscript(contents, index + 1)

    def limitsahead(self, contents, index):
        """Limit the current element based on the next."""
        contents[index + 1].add(contents[index].clone())
        contents[index].output = EmptyOutput()

    def modifylimits(self, contents, index):
        """Modify a limits commands so that the limits appear above and below."""
        limited = contents[index]
        subscript = self.getlimit(contents, index + 1)
        limited.contents.append(subscript)
        if self.checkscript(contents, index + 1):
            superscript = self.getlimit(contents, index + 1)
        else:
            superscript = TaggedBit().constant('\u205f', 'sup class="limit"')
        limited.contents.insert(0, superscript)

    def getlimit(self, contents, index):
        """Get the limit for a limits command."""
        limit = self.getscript(contents, index)
        limit.output.tag = limit.output.tag.replace('script', 'limit')
        return limit

    def modifyscripts(self, contents, index):
        """Modify the super- and subscript to appear vertically aligned."""
        subscript = self.getscript(contents, index)
        superscript = self.getscript(contents, index)
        scripts = TaggedBit().complete([superscript, subscript], 'span class="scripts"')
        contents.insert(index, scripts)

    def checkscript(self, contents, index):
        """Check if the current element is a sub- or superscript."""
        return self.checkcommand(contents, index, SymbolFunction)

    def checkcommand(self, contents, index, type):
        """Check for the given type as the current element."""
        if len(contents) <= index:
            return False
        return isinstance(contents[index], type)

    def getscript(self, contents, index):
        """Get the sub- or superscript."""
        bit = contents[index]
        bit.output.tag += ' class="script"'
        del contents[index]
        return bit