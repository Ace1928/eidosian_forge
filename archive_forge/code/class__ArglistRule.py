import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='arglist')
class _ArglistRule(SyntaxRule):

    @property
    def message(self):
        if self._normalizer.version < (3, 7):
            return 'Generator expression must be parenthesized if not sole argument'
        else:
            return 'Generator expression must be parenthesized'

    def is_issue(self, node):
        arg_set = set()
        kw_only = False
        kw_unpacking_only = False
        for argument in node.children:
            if argument == ',':
                continue
            if argument.type == 'argument':
                first = argument.children[0]
                if _is_argument_comprehension(argument) and len(node.children) >= 2:
                    return True
                if first in ('*', '**'):
                    if first == '*':
                        if kw_unpacking_only:
                            message = 'iterable argument unpacking follows keyword argument unpacking'
                            self.add_issue(argument, message=message)
                    else:
                        kw_unpacking_only = True
                else:
                    kw_only = True
                    if first.type == 'name':
                        if first.value in arg_set:
                            message = 'keyword argument repeated'
                            if self._normalizer.version >= (3, 9):
                                message += ': {}'.format(first.value)
                            self.add_issue(first, message=message)
                        else:
                            arg_set.add(first.value)
            elif kw_unpacking_only:
                message = 'positional argument follows keyword argument unpacking'
                self.add_issue(argument, message=message)
            elif kw_only:
                message = 'positional argument follows keyword argument'
                self.add_issue(argument, message=message)