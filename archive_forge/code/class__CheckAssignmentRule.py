import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
class _CheckAssignmentRule(SyntaxRule):

    def _check_assignment(self, node, is_deletion=False, is_namedexpr=False, is_aug_assign=False):
        error = None
        type_ = node.type
        if type_ == 'lambdef':
            error = 'lambda'
        elif type_ == 'atom':
            first, second = node.children[:2]
            error = _get_comprehension_type(node)
            if error is None:
                if second.type == 'dictorsetmaker':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    elif second.children[1] == ':':
                        error = 'dict display'
                    else:
                        error = 'set display'
                elif first == '{' and second == '}':
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'dict display'
                elif first == '{' and len(node.children) > 2:
                    if self._normalizer.version < (3, 8):
                        error = 'literal'
                    else:
                        error = 'set display'
                elif first in ('(', '['):
                    if second.type == 'yield_expr':
                        error = 'yield expression'
                    elif second.type == 'testlist_comp':
                        if is_namedexpr:
                            if first == '(':
                                error = 'tuple'
                            elif first == '[':
                                error = 'list'
                        for child in second.children[::2]:
                            self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
                    else:
                        self._check_assignment(second, is_deletion, is_namedexpr, is_aug_assign)
        elif type_ == 'keyword':
            if node.value == 'yield':
                error = 'yield expression'
            elif self._normalizer.version < (3, 8):
                error = 'keyword'
            else:
                error = str(node.value)
        elif type_ == 'operator':
            if node.value == '...':
                error = 'Ellipsis'
        elif type_ == 'comparison':
            error = 'comparison'
        elif type_ in ('string', 'number', 'strings'):
            error = 'literal'
        elif type_ == 'yield_expr':
            message = 'assignment to yield expression not possible'
            self.add_issue(node, message=message)
        elif type_ == 'test':
            error = 'conditional expression'
        elif type_ in ('atom_expr', 'power'):
            if node.children[0] == 'await':
                error = 'await expression'
            elif node.children[-2] == '**':
                error = 'operator'
            else:
                trailer = node.children[-1]
                assert trailer.type == 'trailer'
                if trailer.children[0] == '(':
                    error = 'function call'
                elif is_namedexpr and trailer.children[0] == '[':
                    error = 'subscript'
                elif is_namedexpr and trailer.children[0] == '.':
                    error = 'attribute'
        elif type_ == 'fstring':
            if self._normalizer.version < (3, 8):
                error = 'literal'
            else:
                error = 'f-string expression'
        elif type_ in ('testlist_star_expr', 'exprlist', 'testlist'):
            for child in node.children[::2]:
                self._check_assignment(child, is_deletion, is_namedexpr, is_aug_assign)
        elif 'expr' in type_ and type_ != 'star_expr' or '_test' in type_ or type_ in ('term', 'factor'):
            error = 'operator'
        elif type_ == 'star_expr':
            if is_deletion:
                if self._normalizer.version >= (3, 9):
                    error = 'starred'
                else:
                    self.add_issue(node, message="can't use starred expression here")
            else:
                if self._normalizer.version >= (3, 9):
                    ancestor = node.parent
                else:
                    ancestor = _skip_parens_bottom_up(node)
                if ancestor.type not in _STAR_EXPR_PARENTS and (not is_aug_assign) and (not (ancestor.type == 'atom' and ancestor.children[0] == '[')):
                    message = 'starred assignment target must be in a list or tuple'
                    self.add_issue(node, message=message)
            self._check_assignment(node.children[1])
        if error is not None:
            if is_namedexpr:
                message = 'cannot use assignment expressions with %s' % error
            else:
                cannot = "can't" if self._normalizer.version < (3, 8) else 'cannot'
                message = ' '.join([cannot, 'delete' if is_deletion else 'assign to', error])
            self.add_issue(node, message=message)