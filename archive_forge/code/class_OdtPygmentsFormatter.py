import pygments
import pygments.formatter
class OdtPygmentsFormatter(pygments.formatter.Formatter):

    def __init__(self, rststyle_function, escape_function):
        pygments.formatter.Formatter.__init__(self)
        self.rststyle_function = rststyle_function
        self.escape_function = escape_function

    def rststyle(self, name, parameters=()):
        return self.rststyle_function(name, parameters)