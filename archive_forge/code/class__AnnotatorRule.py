import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
@ErrorFinder.register_rule(type='annassign')
class _AnnotatorRule(SyntaxRule):
    message = 'illegal target for annotation'

    def get_node(self, node):
        return node.parent

    def is_issue(self, node):
        type_ = None
        lhs = node.parent.children[0]
        lhs = _remove_parens(lhs)
        try:
            children = lhs.children
        except AttributeError:
            pass
        else:
            if ',' in children or (lhs.type == 'atom' and children[0] == '('):
                type_ = 'tuple'
            elif lhs.type == 'atom' and children[0] == '[':
                type_ = 'list'
            trailer = children[-1]
        if type_ is None:
            if not (lhs.type == 'name' or (lhs.type in ('atom_expr', 'power') and trailer.type == 'trailer' and (trailer.children[0] != '('))):
                return True
        else:
            message = 'only single target (not %s) can be annotated'
            self.add_issue(lhs.parent, message=message % type_)