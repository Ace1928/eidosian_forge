from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class For(Loop):

    def __init__(self, start, condition, update, body):
        super(For, self).__init__(body)
        self.start = start
        self.condition = condition
        self.update = update

    def intro_line(self):
        return 'for (%s; %s; %s)' % (self.start, self.condition, self.update)