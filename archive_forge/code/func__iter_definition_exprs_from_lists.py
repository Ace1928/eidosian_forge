import codecs
import warnings
import re
from contextlib import contextmanager
from parso.normalizer import Normalizer, NormalizerConfig, Issue, Rule
from parso.python.tokenize import _get_token_collection
def _iter_definition_exprs_from_lists(exprlist):

    def check_expr(child):
        if child.type == 'atom':
            if child.children[0] == '(':
                testlist_comp = child.children[1]
                if testlist_comp.type == 'testlist_comp':
                    yield from _iter_definition_exprs_from_lists(testlist_comp)
                    return
                else:
                    yield from check_expr(testlist_comp)
                    return
            elif child.children[0] == '[':
                yield testlist_comp
                return
        yield child
    if exprlist.type in _STAR_EXPR_PARENTS:
        for child in exprlist.children[::2]:
            yield from check_expr(child)
    else:
        yield from check_expr(exprlist)