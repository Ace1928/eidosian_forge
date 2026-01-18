from pyparsing import *
from sys import stdin, argv, exit
class SemanticException(Exception):
    """Exception for semantic errors found during parsing, similar to ParseException.
       Introduced because ParseException is used internally in pyparsing and custom
       messages got lost and replaced by pyparsing's generic errors.
    """

    def __init__(self, message, print_location=True):
        super(SemanticException, self).__init__()
        self._message = message
        self.location = exshared.location
        self.print_location = print_location
        if exshared.location != None:
            self.line = lineno(exshared.location, exshared.text)
            self.col = col(exshared.location, exshared.text)
            self.text = line(exshared.location, exshared.text)
        else:
            self.line = self.col = self.text = None

    def _get_message(self):
        return self._message

    def _set_message(self, message):
        self._message = message
    message = property(_get_message, _set_message)

    def __str__(self):
        """String representation of the semantic error"""
        msg = 'Error'
        if self.print_location and self.line != None:
            msg += ' at line %d, col %d' % (self.line, self.col)
        msg += ': %s' % self.message
        if self.print_location and self.line != None:
            msg += '\n%s' % self.text
        return msg