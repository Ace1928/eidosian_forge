import re
from pygments.lexer import RegexLexer, bygroups, default, words, include
from pygments.token import Comment, Error, Keyword, Name, Number, \
from pygments.lexers import _vbscript_builtins
class BBCBasicLexer(RegexLexer):
    """
    BBC Basic was supplied on the BBC Micro, and later Acorn RISC OS.
    It is also used by BBC Basic For Windows.

    .. versionadded:: 2.4
    """
    base_keywords = ['OTHERWISE', 'AND', 'DIV', 'EOR', 'MOD', 'OR', 'ERROR', 'LINE', 'OFF', 'STEP', 'SPC', 'TAB', 'ELSE', 'THEN', 'OPENIN', 'PTR', 'PAGE', 'TIME', 'LOMEM', 'HIMEM', 'ABS', 'ACS', 'ADVAL', 'ASC', 'ASN', 'ATN', 'BGET', 'COS', 'COUNT', 'DEG', 'ERL', 'ERR', 'EVAL', 'EXP', 'EXT', 'FALSE', 'FN', 'GET', 'INKEY', 'INSTR', 'INT', 'LEN', 'LN', 'LOG', 'NOT', 'OPENUP', 'OPENOUT', 'PI', 'POINT', 'POS', 'RAD', 'RND', 'SGN', 'SIN', 'SQR', 'TAN', 'TO', 'TRUE', 'USR', 'VAL', 'VPOS', 'CHR$', 'GET$', 'INKEY$', 'LEFT$', 'MID$', 'RIGHT$', 'STR$', 'STRING$', 'EOF', 'PTR', 'PAGE', 'TIME', 'LOMEM', 'HIMEM', 'SOUND', 'BPUT', 'CALL', 'CHAIN', 'CLEAR', 'CLOSE', 'CLG', 'CLS', 'DATA', 'DEF', 'DIM', 'DRAW', 'END', 'ENDPROC', 'ENVELOPE', 'FOR', 'GOSUB', 'GOTO', 'GCOL', 'IF', 'INPUT', 'LET', 'LOCAL', 'MODE', 'MOVE', 'NEXT', 'ON', 'VDU', 'PLOT', 'PRINT', 'PROC', 'READ', 'REM', 'REPEAT', 'REPORT', 'RESTORE', 'RETURN', 'RUN', 'STOP', 'COLOUR', 'TRACE', 'UNTIL', 'WIDTH', 'OSCLI']
    basic5_keywords = ['WHEN', 'OF', 'ENDCASE', 'ENDIF', 'ENDWHILE', 'CASE', 'CIRCLE', 'FILL', 'ORIGIN', 'POINT', 'RECTANGLE', 'SWAP', 'WHILE', 'WAIT', 'MOUSE', 'QUIT', 'SYS', 'INSTALL', 'LIBRARY', 'TINT', 'ELLIPSE', 'BEATS', 'TEMPO', 'VOICES', 'VOICE', 'STEREO', 'OVERLAY', 'APPEND', 'AUTO', 'CRUNCH', 'DELETE', 'EDIT', 'HELP', 'LIST', 'LOAD', 'LVAR', 'NEW', 'OLD', 'RENUMBER', 'SAVE', 'TEXTLOAD', 'TEXTSAVE', 'TWIN', 'TWINO', 'INSTALL', 'SUM', 'BEAT']
    name = 'BBC Basic'
    aliases = ['bbcbasic']
    filenames = ['*.bbc']
    tokens = {'root': [('[0-9]+', Name.Label), ('(\\*)([^\\n]*)', bygroups(Keyword.Pseudo, Comment.Special)), default('code')], 'code': [('(REM)([^\\n]*)', bygroups(Keyword.Declaration, Comment.Single)), ('\\n', Whitespace, 'root'), ('\\s+', Whitespace), (':', Comment.Preproc), ('(DEF)(\\s*)(FN|PROC)([A-Za-z_@][\\w@]*)', bygroups(Keyword.Declaration, Whitespace, Keyword.Declaration, Name.Function)), ('(FN|PROC)([A-Za-z_@][\\w@]*)', bygroups(Keyword, Name.Function)), ('(GOTO|GOSUB|THEN|RESTORE)(\\s*)(\\d+)', bygroups(Keyword, Whitespace, Name.Label)), ('(TRUE|FALSE)', Keyword.Constant), ('(PAGE|LOMEM|HIMEM|TIME|WIDTH|ERL|ERR|REPORT\\$|POS|VPOS|VOICES)', Keyword.Pseudo), (words(base_keywords), Keyword), (words(basic5_keywords), Keyword), ('"', String.Double, 'string'), ('%[01]{1,32}', Number.Bin), ('&[0-9a-f]{1,8}', Number.Hex), ('[+-]?[0-9]+\\.[0-9]*(E[+-]?[0-9]+)?', Number.Float), ('[+-]?\\.[0-9]+(E[+-]?[0-9]+)?', Number.Float), ('[+-]?[0-9]+E[+-]?[0-9]+', Number.Float), ('[+-]?\\d+', Number.Integer), ('([A-Za-z_@][\\w@]*[%$]?)', Name.Variable), ('([+\\-]=|[$!|?+\\-*/%^=><();]|>=|<=|<>|<<|>>|>>>|,)', Operator)], 'string': [('[^"\\n]+', String.Double), ('"', String.Double, '#pop'), ('\\n', Error, 'root')]}

    def analyse_text(text):
        if text.startswith('10REM >') or text.startswith('REM >'):
            return 0.9