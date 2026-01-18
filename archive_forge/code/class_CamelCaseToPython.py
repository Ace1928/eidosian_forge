import functools
import re
class CamelCaseToPython:
    """ Simple functor class to convert names from camel case to idiomatic
    Python variable names.

    For example::
        >>> camel2python = CamelCaseToPython
        >>> camel2python('XMLActor2DToSGML')
        'xml_actor2d_to_sgml'
    """

    def __init__(self):
        self.patn = re.compile('([A-Z0-9]+)([a-z0-9]*)')
        self.nd_patn = re.compile('(\\D[123])_D')

    def __call__(self, name):
        ret = self.patn.sub(self._repl, name)
        ret = self.nd_patn.sub('\\1d', ret)
        if ret[0] == '_':
            ret = ret[1:]
        return ret.lower()

    def _repl(self, m):
        g1 = m.group(1)
        g2 = m.group(2)
        if len(g1) > 1:
            if g2:
                return '_' + g1[:-1] + '_' + g1[-1] + g2
            else:
                return '_' + g1
        else:
            return '_' + g1 + g2