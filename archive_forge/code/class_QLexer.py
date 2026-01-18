from pygments.lexer import RegexLexer, words, include, bygroups, inherit
from pygments.token import Comment, Name, Number, Operator, Punctuation, \
class QLexer(KLexer):
    """
    For `Q <https://code.kx.com/>`_ source code.

    .. versionadded:: 2.12
    """
    name = 'Q'
    aliases = ['q']
    filenames = ['*.q']
    tokens = {'root': [(words(('aj', 'aj0', 'ajf', 'ajf0', 'all', 'and', 'any', 'asc', 'asof', 'attr', 'avgs', 'ceiling', 'cols', 'count', 'cross', 'csv', 'cut', 'deltas', 'desc', 'differ', 'distinct', 'dsave', 'each', 'ej', 'ema', 'eval', 'except', 'fby', 'fills', 'first', 'fkeys', 'flip', 'floor', 'get', 'group', 'gtime', 'hclose', 'hcount', 'hdel', 'hsym', 'iasc', 'idesc', 'ij', 'ijf', 'inter', 'inv', 'key', 'keys', 'lj', 'ljf', 'load', 'lower', 'lsq', 'ltime', 'ltrim', 'mavg', 'maxs', 'mcount', 'md5', 'mdev', 'med', 'meta', 'mins', 'mmax', 'mmin', 'mmu', 'mod', 'msum', 'neg', 'next', 'not', 'null', 'or', 'over', 'parse', 'peach', 'pj', 'prds', 'prior', 'prev', 'rand', 'rank', 'ratios', 'raze', 'read0', 'read1', 'reciprocal', 'reval', 'reverse', 'rload', 'rotate', 'rsave', 'rtrim', 'save', 'scan', 'scov', 'sdev', 'set', 'show', 'signum', 'ssr', 'string', 'sublist', 'sums', 'sv', 'svar', 'system', 'tables', 'til', 'trim', 'txf', 'type', 'uj', 'ujf', 'ungroup', 'union', 'upper', 'upsert', 'value', 'view', 'views', 'vs', 'where', 'wj', 'wj1', 'ww', 'xasc', 'xbar', 'xcol', 'xcols', 'xdesc', 'xgroup', 'xkey', 'xlog', 'xprev', 'xrank'), suffix='\\b'), Name.Builtin), inherit]}