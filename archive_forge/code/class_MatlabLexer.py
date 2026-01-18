import re
from pygments.lexer import Lexer, RegexLexer, bygroups, words, do_insertions
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
from pygments.lexers import _scilab_builtins
class MatlabLexer(RegexLexer):
    """
    For Matlab source code.

    .. versionadded:: 0.10
    """
    name = 'Matlab'
    aliases = ['matlab']
    filenames = ['*.m']
    mimetypes = ['text/matlab']
    elfun = ('sin', 'sind', 'sinh', 'asin', 'asind', 'asinh', 'cos', 'cosd', 'cosh', 'acos', 'acosd', 'acosh', 'tan', 'tand', 'tanh', 'atan', 'atand', 'atan2', 'atanh', 'sec', 'secd', 'sech', 'asec', 'asecd', 'asech', 'csc', 'cscd', 'csch', 'acsc', 'acscd', 'acsch', 'cot', 'cotd', 'coth', 'acot', 'acotd', 'acoth', 'hypot', 'exp', 'expm1', 'log', 'log1p', 'log10', 'log2', 'pow2', 'realpow', 'reallog', 'realsqrt', 'sqrt', 'nthroot', 'nextpow2', 'abs', 'angle', 'complex', 'conj', 'imag', 'real', 'unwrap', 'isreal', 'cplxpair', 'fix', 'floor', 'ceil', 'round', 'mod', 'rem', 'sign')
    specfun = ('airy', 'besselj', 'bessely', 'besselh', 'besseli', 'besselk', 'beta', 'betainc', 'betaln', 'ellipj', 'ellipke', 'erf', 'erfc', 'erfcx', 'erfinv', 'expint', 'gamma', 'gammainc', 'gammaln', 'psi', 'legendre', 'cross', 'dot', 'factor', 'isprime', 'primes', 'gcd', 'lcm', 'rat', 'rats', 'perms', 'nchoosek', 'factorial', 'cart2sph', 'cart2pol', 'pol2cart', 'sph2cart', 'hsv2rgb', 'rgb2hsv')
    elmat = ('zeros', 'ones', 'eye', 'repmat', 'rand', 'randn', 'linspace', 'logspace', 'freqspace', 'meshgrid', 'accumarray', 'size', 'length', 'ndims', 'numel', 'disp', 'isempty', 'isequal', 'isequalwithequalnans', 'cat', 'reshape', 'diag', 'blkdiag', 'tril', 'triu', 'fliplr', 'flipud', 'flipdim', 'rot90', 'find', 'end', 'sub2ind', 'ind2sub', 'bsxfun', 'ndgrid', 'permute', 'ipermute', 'shiftdim', 'circshift', 'squeeze', 'isscalar', 'isvector', 'ans', 'eps', 'realmax', 'realmin', 'pi', 'i', 'inf', 'nan', 'isnan', 'isinf', 'isfinite', 'j', 'why', 'compan', 'gallery', 'hadamard', 'hankel', 'hilb', 'invhilb', 'magic', 'pascal', 'rosser', 'toeplitz', 'vander', 'wilkinson')
    tokens = {'root': [('^!.*', String.Other), ('%\\{\\s*\\n', Comment.Multiline, 'blockcomment'), ('%.*$', Comment), ('^\\s*function', Keyword, 'deffunc'), (words(('break', 'case', 'catch', 'classdef', 'continue', 'else', 'elseif', 'end', 'enumerated', 'events', 'for', 'function', 'global', 'if', 'methods', 'otherwise', 'parfor', 'persistent', 'properties', 'return', 'spmd', 'switch', 'try', 'while'), suffix='\\b'), Keyword), ('(' + '|'.join(elfun + specfun + elmat) + ')\\b', Name.Builtin), ('\\.\\.\\..*$', Comment), ('-|==|~=|<|>|<=|>=|&&|&|~|\\|\\|?', Operator), ('\\.\\*|\\*|\\+|\\.\\^|\\.\\\\|\\.\\/|\\/|\\\\', Operator), ('\\[|\\]|\\(|\\)|\\{|\\}|:|@|\\.|,', Punctuation), ('=|:|;', Punctuation), ("(?<=[\\w)\\].])\\'+", Operator), ('(\\d+\\.\\d*|\\d*\\.\\d+)([eEf][+-]?[0-9]+)?', Number.Float), ('\\d+[eEf][+-]?[0-9]+', Number.Float), ('\\d+', Number.Integer), ("(?<![\\w)\\].])\\'", String, 'string'), ('[a-zA-Z_]\\w*', Name), ('.', Text)], 'string': [("[^\\']*\\'", String, '#pop')], 'blockcomment': [('^\\s*%\\}', Comment.Multiline, '#pop'), ('^.*\\n', Comment.Multiline), ('.', Comment.Multiline)], 'deffunc': [('(\\s*)(?:(.+)(\\s*)(=)(\\s*))?(.+)(\\()(.*)(\\))(\\s*)', bygroups(Whitespace, Text, Whitespace, Punctuation, Whitespace, Name.Function, Punctuation, Text, Punctuation, Whitespace), '#pop'), ('(\\s*)([a-zA-Z_]\\w*)', bygroups(Text, Name.Function), '#pop')]}

    def analyse_text(text):
        if re.match('^\\s*%', text, re.M):
            return 0.2
        elif re.match('^!\\w+', text, re.M):
            return 0.2