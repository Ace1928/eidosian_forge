from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class Typedef(DeclSpecifier):

    def __init__(self, subdecl):
        DeclSpecifier.__init__(self, subdecl, 'typedef')