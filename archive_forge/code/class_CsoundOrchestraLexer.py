import re
from pygments.lexer import RegexLexer, bygroups, default, include, using, words
from pygments.token import Comment, Keyword, Name, Number, Operator, Punctuation, \
from pygments.lexers._csound_builtins import OPCODES
from pygments.lexers.html import HtmlLexer
from pygments.lexers.python import PythonLexer
from pygments.lexers.scripting import LuaLexer
class CsoundOrchestraLexer(CsoundLexer):
    """
    For `Csound <http://csound.github.io>`_ orchestras.

    .. versionadded:: 2.1
    """
    name = 'Csound Orchestra'
    aliases = ['csound', 'csound-orc']
    filenames = ['*.orc']
    user_defined_opcodes = set()

    def opcode_name_callback(lexer, match):
        opcode = match.group(0)
        lexer.user_defined_opcodes.add(opcode)
        yield (match.start(), Name.Function, opcode)

    def name_callback(lexer, match):
        name = match.group(0)
        if re.match('p\\d+$', name) or name in OPCODES:
            yield (match.start(), Name.Builtin, name)
        elif name in lexer.user_defined_opcodes:
            yield (match.start(), Name.Function, name)
        else:
            nameMatch = re.search('^(g?[aikSw])(\\w+)', name)
            if nameMatch:
                yield (nameMatch.start(1), Keyword.Type, nameMatch.group(1))
                yield (nameMatch.start(2), Name, nameMatch.group(2))
            else:
                yield (match.start(), Name, name)
    tokens = {'label': [('\\b(\\w+)(:)', bygroups(Name.Label, Punctuation))], 'partial expression': [include('preprocessor directives'), ('\\b(0dbfs|k(r|smps)|nchnls(_i)?|sr)\\b', Name.Variable.Global), ('\\d+e[+-]?\\d+|(\\d+\\.\\d*|\\d*\\.\\d+)(e[+-]?\\d+)?', Number.Float), ('0[xX][a-fA-F0-9]+', Number.Hex), ('\\d+', Number.Integer), ('"', String, 'single-line string'), ('\\{\\{', String, 'multi-line string'), ('[+\\-*/%^!=&|<>#~¬]', Operator), ('[](),?:[]', Punctuation), (words(('do', 'else', 'elseif', 'endif', 'enduntil', 'fi', 'if', 'ithen', 'kthen', 'od', 'then', 'until', 'while', 'return', 'timout'), prefix='\\b', suffix='\\b'), Keyword), (words(('goto', 'igoto', 'kgoto', 'rigoto', 'tigoto'), prefix='\\b', suffix='\\b'), Keyword, 'goto label'), (words(('cggoto', 'cigoto', 'cingoto', 'ckgoto', 'cngoto'), prefix='\\b', suffix='\\b'), Keyword, ('goto label', 'goto expression')), (words(('loop_ge', 'loop_gt', 'loop_le', 'loop_lt'), prefix='\\b', suffix='\\b'), Keyword, ('goto label', 'goto expression', 'goto expression', 'goto expression')), ('\\bscoreline(_i)?\\b', Name.Builtin, 'scoreline opcode'), ('\\bpyl?run[it]?\\b', Name.Builtin, 'python opcode'), ('\\blua_(exec|opdef)\\b', Name.Builtin, 'lua opcode'), ('\\b[a-zA-Z_]\\w*\\b', name_callback)], 'expression': [include('whitespace or macro call'), newline + ('#pop',), include('partial expression')], 'root': [newline, include('whitespace or macro call'), ('\\binstr\\b', Keyword, ('instrument block', 'instrument name list')), ('\\bopcode\\b', Keyword, ('opcode block', 'opcode parameter list', 'opcode types', 'opcode types', 'opcode name')), include('label'), default('expression')], 'instrument name list': [include('whitespace or macro call'), ('\\d+|\\+?[a-zA-Z_]\\w*', Name.Function), (',', Punctuation), newline + ('#pop',)], 'instrument block': [newline, include('whitespace or macro call'), ('\\bendin\\b', Keyword, '#pop'), include('label'), default('expression')], 'opcode name': [include('whitespace or macro call'), ('[a-zA-Z_]\\w*', opcode_name_callback, '#pop')], 'opcode types': [include('whitespace or macro call'), ('0|[]afijkKoOpPStV[]+', Keyword.Type, '#pop'), (',', Punctuation)], 'opcode parameter list': [include('whitespace or macro call'), newline + ('#pop',)], 'opcode block': [newline, include('whitespace or macro call'), ('\\bendop\\b', Keyword, '#pop'), include('label'), default('expression')], 'goto label': [include('whitespace or macro call'), ('\\w+', Name.Label, '#pop'), default('#pop')], 'goto expression': [include('whitespace or macro call'), (',', Punctuation, '#pop'), include('partial expression')], 'single-line string': [include('macro call'), ('"', String, '#pop'), ('%\\d*(\\.\\d+)?[cdhilouxX]', String.Interpol), ('%[!%nNrRtT]|[~^]|\\\\([\\\\aAbBnNrRtT"]|[0-7]{1,3})', String.Escape), ('[^\\\\"~$%\\^\\n]+', String), ('[\\\\"~$%\\^\\n]', String)], 'multi-line string': [('\\}\\}', String, '#pop'), ('[^}]+|\\}(?!\\})', String)], 'scoreline opcode': [include('whitespace or macro call'), ('\\{\\{', String, 'scoreline'), default('#pop')], 'scoreline': [('\\}\\}', String, '#pop'), ('([^}]+)|\\}(?!\\})', using(CsoundScoreLexer))], 'python opcode': [include('whitespace or macro call'), ('\\{\\{', String, 'python'), default('#pop')], 'python': [('\\}\\}', String, '#pop'), ('([^}]+)|\\}(?!\\})', using(PythonLexer))], 'lua opcode': [include('whitespace or macro call'), ('"', String, 'single-line string'), ('\\{\\{', String, 'lua'), (',', Punctuation), default('#pop')], 'lua': [('\\}\\}', String, '#pop'), ('([^}]+)|\\}(?!\\})', using(LuaLexer))]}