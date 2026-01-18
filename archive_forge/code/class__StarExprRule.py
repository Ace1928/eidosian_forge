import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='star_expr')
class _StarExprRule(SyntaxRule):
    message_iterable_unpacking = 'iterable unpacking cannot be used in comprehension'

    def is_issue(self, node):

        def check_delete_starred(node):
            while node.parent is not None:
                node = node.parent
                if node.type == 'del_stmt':
                    return True
                if node.type not in (*_STAR_EXPR_PARENTS, 'atom'):
                    return False
            return False
        if self._normalizer.version >= (3, 9):
            ancestor = node.parent
        else:
            ancestor = _skip_parens_bottom_up(node)
        if ancestor.type not in (*_STAR_EXPR_PARENTS, 'dictorsetmaker') and (not (ancestor.type == 'atom' and ancestor.children[0] != '(')):
            self.add_issue(node, message="can't use starred expression here")
            return
        if check_delete_starred(node):
            if self._normalizer.version >= (3, 9):
                self.add_issue(node, message='cannot delete starred')
            else:
                self.add_issue(node, message="can't use starred expression here")
            return
        if node.parent.type == 'testlist_comp':
            if node.parent.children[1].type in _COMP_FOR_TYPES:
                self.add_issue(node, message=self.message_iterable_unpacking)