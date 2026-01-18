import re
from pygments.lexer import RegexLexer, include, bygroups, using, default
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers.html import HtmlLexer
from pygments.lexers import _stan_builtins
class BugsLexer(RegexLexer):
    """
    Pygments Lexer for `OpenBugs <http://www.openbugs.net/>`_ and WinBugs
    models.

    .. versionadded:: 1.6
    """
    name = 'BUGS'
    aliases = ['bugs', 'winbugs', 'openbugs']
    filenames = ['*.bug']
    _FUNCTIONS = ('abs', 'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctanh', 'cloglog', 'cos', 'cosh', 'cumulative', 'cut', 'density', 'deviance', 'equals', 'expr', 'gammap', 'ilogit', 'icloglog', 'integral', 'log', 'logfact', 'loggam', 'logit', 'max', 'min', 'phi', 'post.p.value', 'pow', 'prior.p.value', 'probit', 'replicate.post', 'replicate.prior', 'round', 'sin', 'sinh', 'solution', 'sqrt', 'step', 'tan', 'tanh', 'trunc', 'inprod', 'interp.lin', 'inverse', 'logdet', 'mean', 'eigen.vals', 'ode', 'prod', 'p.valueM', 'rank', 'ranked', 'replicate.postM', 'sd', 'sort', 'sum', 'D', 'I', 'F', 'T', 'C')
    ' OpenBUGS built-in functions\n\n    From http://www.openbugs.info/Manuals/ModelSpecification.html#ContentsAII\n\n    This also includes\n\n    - T, C, I : Truncation and censoring.\n      ``T`` and ``C`` are in OpenBUGS. ``I`` in WinBUGS.\n    - D : ODE\n    - F : Functional http://www.openbugs.info/Examples/Functionals.html\n\n    '
    _DISTRIBUTIONS = ('dbern', 'dbin', 'dcat', 'dnegbin', 'dpois', 'dhyper', 'dbeta', 'dchisqr', 'ddexp', 'dexp', 'dflat', 'dgamma', 'dgev', 'df', 'dggamma', 'dgpar', 'dloglik', 'dlnorm', 'dlogis', 'dnorm', 'dpar', 'dt', 'dunif', 'dweib', 'dmulti', 'ddirch', 'dmnorm', 'dmt', 'dwish')
    ' OpenBUGS built-in distributions\n\n    Functions from\n    http://www.openbugs.info/Manuals/ModelSpecification.html#ContentsAI\n    '
    tokens = {'whitespace': [('\\s+', Text)], 'comments': [('#.*$', Comment.Single)], 'root': [include('comments'), include('whitespace'), ('(model)(\\s+)(\\{)', bygroups(Keyword.Namespace, Text, Punctuation)), ('(for|in)(?![\\w.])', Keyword.Reserved), ('(%s)(?=\\s*\\()' % '|'.join(_FUNCTIONS + _DISTRIBUTIONS), Name.Builtin), ('[A-Za-z][\\w.]*', Name), ('[-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?', Number), ('\\[|\\]|\\(|\\)|:|,|;', Punctuation), ('<-|~', Operator), ('\\+|-|\\*|/', Operator), ('[{}]', Punctuation)]}

    def analyse_text(text):
        if re.search('^\\s*model\\s*{', text, re.M):
            return 0.7
        else:
            return 0.0