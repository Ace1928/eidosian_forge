import functools
import re
import nltk.tree
def _tgrep_relation_action(_s, _l, tokens):
    """
    Builds a lambda function representing a predicate on a tree node
    depending on its relation to other nodes in the tree.
    """
    negated = False
    if tokens[0] == '!':
        negated = True
        tokens = tokens[1:]
    if tokens[0] == '[':
        assert len(tokens) == 3
        assert tokens[2] == ']'
        retval = tokens[1]
    else:
        assert len(tokens) == 2
        operator, predicate = tokens
        if operator == '<':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in n))
        elif operator == '>':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and predicate(n.parent(), m, l)
        elif operator == '<,' or operator == '<1':
            retval = lambda n, m=None, l=None: _istree(n) and bool(list(n)) and predicate(n[0], m, l)
        elif operator == '>,' or operator == '>1':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (n is n.parent()[0]) and predicate(n.parent(), m, l)
        elif operator[0] == '<' and operator[1:].isdigit():
            idx = int(operator[1:])
            retval = (lambda i: lambda n, m=None, l=None: _istree(n) and bool(list(n)) and (0 <= i < len(n)) and predicate(n[i], m, l))(idx - 1)
        elif operator[0] == '>' and operator[1:].isdigit():
            idx = int(operator[1:])
            retval = (lambda i: lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (0 <= i < len(n.parent())) and (n is n.parent()[i]) and predicate(n.parent(), m, l))(idx - 1)
        elif operator == "<'" or operator == '<-' or operator == '<-1':
            retval = lambda n, m=None, l=None: _istree(n) and bool(list(n)) and predicate(n[-1], m, l)
        elif operator == ">'" or operator == '>-' or operator == '>-1':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (n is n.parent()[-1]) and predicate(n.parent(), m, l)
        elif operator[:2] == '<-' and operator[2:].isdigit():
            idx = -int(operator[2:])
            retval = (lambda i: lambda n, m=None, l=None: _istree(n) and bool(list(n)) and (0 <= i + len(n) < len(n)) and predicate(n[i + len(n)], m, l))(idx)
        elif operator[:2] == '>-' and operator[2:].isdigit():
            idx = -int(operator[2:])
            retval = (lambda i: lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (0 <= i + len(n.parent()) < len(n.parent())) and (n is n.parent()[i + len(n.parent())]) and predicate(n.parent(), m, l))(idx)
        elif operator == '<:':
            retval = lambda n, m=None, l=None: _istree(n) and len(n) == 1 and predicate(n[0], m, l)
        elif operator == '>:':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and (len(n.parent()) == 1) and predicate(n.parent(), m, l)
        elif operator == '<<':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _descendants(n)))
        elif operator == '>>':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in ancestors(n)))
        elif operator == '<<,' or operator == '<<1':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _leftmost_descendants(n)))
        elif operator == '>>,':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) and n in _leftmost_descendants(x) for x in ancestors(n)))
        elif operator == "<<'":
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _rightmost_descendants(n)))
        elif operator == ">>'":
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) and n in _rightmost_descendants(x) for x in ancestors(n)))
        elif operator == '<<:':
            retval = lambda n, m=None, l=None: _istree(n) and any((predicate(x, m, l) for x in _unique_descendants(n)))
        elif operator == '>>:':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in unique_ancestors(n)))
        elif operator == '.':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _immediately_after(n)))
        elif operator == ',':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _immediately_before(n)))
        elif operator == '..':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _after(n)))
        elif operator == ',,':
            retval = lambda n, m=None, l=None: any((predicate(x, m, l) for x in _before(n)))
        elif operator == '$' or operator == '%':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent() if x is not n))
        elif operator == '$.' or operator == '%.':
            retval = lambda n, m=None, l=None: hasattr(n, 'right_sibling') and bool(n.right_sibling()) and predicate(n.right_sibling(), m, l)
        elif operator == '$,' or operator == '%,':
            retval = lambda n, m=None, l=None: hasattr(n, 'left_sibling') and bool(n.left_sibling()) and predicate(n.left_sibling(), m, l)
        elif operator == '$..' or operator == '%..':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and hasattr(n, 'parent_index') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent()[n.parent_index() + 1:]))
        elif operator == '$,,' or operator == '%,,':
            retval = lambda n, m=None, l=None: hasattr(n, 'parent') and hasattr(n, 'parent_index') and bool(n.parent()) and any((predicate(x, m, l) for x in n.parent()[:n.parent_index()]))
        else:
            raise TgrepException(f'cannot interpret tgrep operator "{operator}"')
    if negated:
        return (lambda r: lambda n, m=None, l=None: not r(n, m, l))(retval)
    else:
        return retval