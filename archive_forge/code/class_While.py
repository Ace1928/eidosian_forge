from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class While(Loop):

    def __init__(self, condition, body):
        super(While, self).__init__(body)
        self.condition = condition

    def intro_line(self):
        return 'while (%s)' % self.condition