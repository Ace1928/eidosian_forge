import sys
import yaql
from yaql.language import exceptions
from yaql.language import utils
class ListExpression(Function):

    def __init__(self, *args):
        super(ListExpression, self).__init__('#list', *args)
        self.uses_receiver = False