from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class FunctionBody(object):

    def __init__(self, fdecl, body):
        """Initialize a function definition. *fdecl* is expected to be
        a :class:`FunctionDeclaration` instance, while *body* is a
        :class:`Block`.
        """
        self.fdecl = fdecl
        self.body = body

    def generate(self):
        for f_line in self.fdecl.generate(with_semicolon=False):
            yield f_line
        for b_line in self.body.generate():
            yield b_line