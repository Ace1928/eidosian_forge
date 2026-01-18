from pygments.lexer import RegexLexer, words, include, bygroups, using, \
from pygments.token import Text, Comment, Operator, Keyword, Name, \
class PostScriptLexer(RegexLexer):
    """
    Lexer for PostScript files.

    The PostScript Language Reference published by Adobe at
    <http://partners.adobe.com/public/developer/en/ps/PLRM.pdf>
    is the authority for this.

    .. versionadded:: 1.4
    """
    name = 'PostScript'
    aliases = ['postscript', 'postscr']
    filenames = ['*.ps', '*.eps']
    mimetypes = ['application/postscript']
    delimiter = '()<>\\[\\]{}/%\\s'
    delimiter_end = '(?=[%s])' % delimiter
    valid_name_chars = '[^%s]' % delimiter
    valid_name = '%s+%s' % (valid_name_chars, delimiter_end)
    tokens = {'root': [('^%!.+\\n', Comment.Preproc), ('%%.*\\n', Comment.Special), ('(^%.*\\n){2,}', Comment.Multiline), ('%.*\\n', Comment.Single), ('\\(', String, 'stringliteral'), ('[{}<>\\[\\]]', Punctuation), ('<[0-9A-Fa-f]+>' + delimiter_end, Number.Hex), ('[0-9]+\\#(\\-|\\+)?([0-9]+\\.?|[0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)((e|E)[0-9]+)?' + delimiter_end, Number.Oct), ('(\\-|\\+)?([0-9]+\\.?|[0-9]*\\.[0-9]+|[0-9]+\\.[0-9]*)((e|E)[0-9]+)?' + delimiter_end, Number.Float), ('(\\-|\\+)?[0-9]+' + delimiter_end, Number.Integer), ('\\/%s' % valid_name, Name.Variable), (valid_name, Name.Function), ('(false|true)' + delimiter_end, Keyword.Constant), ('(eq|ne|g[et]|l[et]|and|or|not|if(?:else)?|for(?:all)?)' + delimiter_end, Keyword.Reserved), (words(('abs', 'add', 'aload', 'arc', 'arcn', 'array', 'atan', 'begin', 'bind', 'ceiling', 'charpath', 'clip', 'closepath', 'concat', 'concatmatrix', 'copy', 'cos', 'currentlinewidth', 'currentmatrix', 'currentpoint', 'curveto', 'cvi', 'cvs', 'def', 'defaultmatrix', 'dict', 'dictstackoverflow', 'div', 'dtransform', 'dup', 'end', 'exch', 'exec', 'exit', 'exp', 'fill', 'findfont', 'floor', 'get', 'getinterval', 'grestore', 'gsave', 'gt', 'identmatrix', 'idiv', 'idtransform', 'index', 'invertmatrix', 'itransform', 'length', 'lineto', 'ln', 'load', 'log', 'loop', 'matrix', 'mod', 'moveto', 'mul', 'neg', 'newpath', 'pathforall', 'pathbbox', 'pop', 'print', 'pstack', 'put', 'quit', 'rand', 'rangecheck', 'rcurveto', 'repeat', 'restore', 'rlineto', 'rmoveto', 'roll', 'rotate', 'round', 'run', 'save', 'scale', 'scalefont', 'setdash', 'setfont', 'setgray', 'setlinecap', 'setlinejoin', 'setlinewidth', 'setmatrix', 'setrgbcolor', 'shfill', 'show', 'showpage', 'sin', 'sqrt', 'stack', 'stringwidth', 'stroke', 'strokepath', 'sub', 'syntaxerror', 'transform', 'translate', 'truncate', 'typecheck', 'undefined', 'undefinedfilename', 'undefinedresult'), suffix=delimiter_end), Name.Builtin), ('\\s+', Text)], 'stringliteral': [('[^()\\\\]+', String), ('\\\\', String.Escape, 'escape'), ('\\(', String, '#push'), ('\\)', String, '#pop')], 'escape': [('[0-8]{3}|n|r|t|b|f|\\\\|\\(|\\)', String.Escape, '#pop'), default('#pop')]}