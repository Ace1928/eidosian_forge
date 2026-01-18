from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class AnnotatedStatement(object):

    def __init__(self, stmt, annotations):
        self.stmt = stmt
        self.annotations = annotations

    def generate(self):
        for directive in self.annotations:
            pragma = '#pragma ' + directive.s
            yield pragma.format(*directive.deps)
        for s in self.stmt.generate():
            yield s