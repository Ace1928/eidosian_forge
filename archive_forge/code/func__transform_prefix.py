from __future__ import unicode_literals
import re
from six.moves import range
from .regex_parser import Any, Sequence, Regex, Variable, Repeat, Lookahead
from .regex_parser import parse_regex, tokenize_regex
@classmethod
def _transform_prefix(cls, root_node, create_group_func):
    """
        Yield all the regular expressions matching a prefix of the grammar
        defined by the `Node` instance.

        This can yield multiple expressions, because in the case of on OR
        operation in the grammar, we can have another outcome depending on
        which clause would appear first. E.g. "(A|B)C" is not the same as
        "(B|A)C" because the regex engine is lazy and takes the first match.
        However, because we the current input is actually a prefix of the
        grammar which meight not yet contain the data for "C", we need to know
        both intermediate states, in order to call the appropriate
        autocompletion for both cases.

        :param root_node: The :class:`Node` instance for which we generate the grammar.
        :param create_group_func: A callable which takes a `Node` and returns the next
            free name for this node.
        """

    def transform(node):
        if isinstance(node, Any):
            for c in node.children:
                for r in transform(c):
                    yield ('(?:%s)?' % r)
        elif isinstance(node, Sequence):
            for i in range(len(node.children)):
                a = [cls._transform(c, create_group_func) for c in node.children[:i]]
                for c in transform(node.children[i]):
                    yield ('(?:%s)' % (''.join(a) + c))
        elif isinstance(node, Regex):
            yield ('(?:%s)?' % node.regex)
        elif isinstance(node, Lookahead):
            if node.negative:
                yield ('(?!%s)' % cls._transform(node.childnode, create_group_func))
            else:
                raise Exception('Positive lookahead not yet supported.')
        elif isinstance(node, Variable):
            for c in transform(node.childnode):
                yield ('(?P<%s>%s)' % (create_group_func(node), c))
        elif isinstance(node, Repeat):
            prefix = cls._transform(node.childnode, create_group_func)
            for c in transform(node.childnode):
                if node.max_repeat:
                    repeat_sign = '{,%i}' % (node.max_repeat - 1)
                else:
                    repeat_sign = '*'
                yield ('(?:%s)%s%s(?:%s)?' % (prefix, repeat_sign, '' if node.greedy else '?', c))
        else:
            raise TypeError('Got %r' % node)
    for r in transform(root_node):
        yield ('^%s$' % r)