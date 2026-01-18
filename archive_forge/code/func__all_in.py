import ast
import operator
import pyparsing
def _all_in(x, *y):
    x = ast.literal_eval(x)
    if not isinstance(x, list):
        raise TypeError('<all-in> must compare with a list literal string, EG "%s"' % (['aes', 'mmx'],))
    return all((val in x for val in y))