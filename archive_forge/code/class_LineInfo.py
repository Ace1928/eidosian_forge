from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class LineInfo(object):

    def __init__(self, filepath, lineno):
        self.filepath = filepath
        self.lineno = lineno

    def generate(self):
        if self.filename and self.lineno:
            yield '#line {} {}'.format(self.lineno, self.filepath)