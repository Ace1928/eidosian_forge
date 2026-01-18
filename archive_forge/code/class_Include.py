from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class Include(object):

    def __init__(self, filename, system=True):
        self.filename = filename
        self.system = system

    def generate(self):
        if self.system:
            yield ('#include <%s>' % self.filename)
        else:
            yield ('#include "%s"' % self.filename)