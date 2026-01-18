from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class ReturnStatement(Statement):

    def generate(self):
        yield ('return ' + self.text + ';')