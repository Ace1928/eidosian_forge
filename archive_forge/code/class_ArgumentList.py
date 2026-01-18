import re
from mako import exceptions
from mako import pyparser
class ArgumentList:
    """parses a fragment of code as a comma-separated list of expressions"""

    def __init__(self, code, **exception_kwargs):
        self.codeargs = []
        self.args = []
        self.declared_identifiers = set()
        self.undeclared_identifiers = set()
        if isinstance(code, str):
            if re.match('\\S', code) and (not re.match(',\\s*$', code)):
                code += ','
            expr = pyparser.parse(code, 'exec', **exception_kwargs)
        else:
            expr = code
        f = pyparser.FindTuple(self, PythonCode, **exception_kwargs)
        f.visit(expr)