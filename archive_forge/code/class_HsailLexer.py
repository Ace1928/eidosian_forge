import re
from pygments.lexer import RegexLexer, include, bygroups, using, words, \
from pygments.lexers.c_cpp import CppLexer, CLexer
from pygments.lexers.d import DLexer
from pygments.token import Text, Name, Number, String, Comment, Punctuation, \
class HsailLexer(RegexLexer):
    """
    For HSAIL assembly code.

    .. versionadded:: 2.2
    """
    name = 'HSAIL'
    aliases = ['hsail', 'hsa']
    filenames = ['*.hsail']
    mimetypes = ['text/x-hsail']
    string = '"[^"]*?"'
    identifier = '[a-zA-Z_][\\w.]*'
    register_number = '[0-9]+'
    register = '(\\$(c|s|d|q)' + register_number + ')'
    alignQual = '(align\\(\\d+\\))'
    widthQual = '(width\\((\\d+|all)\\))'
    allocQual = '(alloc\\(agent\\))'
    roundingMod = '((_ftz)?(_up|_down|_zero|_near))'
    datatypeMod = '_(u8x4|s8x4|u16x2|s16x2|u8x8|s8x8|u16x4|s16x4|u32x2|s32x2|u8x16|s8x16|u16x8|s16x8|u32x4|s32x4|u64x2|s64x2|f16x2|f16x4|f16x8|f32x2|f32x4|f64x2|u8|s8|u16|s16|u32|s32|u64|s64|b128|b8|b16|b32|b64|b1|f16|f32|f64|roimg|woimg|rwimg|samp|sig32|sig64)'
    float = '((\\d+\\.)|(\\d*\\.\\d+))[eE][+-]?\\d+'
    hexfloat = '0[xX](([0-9a-fA-F]+\\.[0-9a-fA-F]*)|([0-9a-fA-F]*\\.[0-9a-fA-F]+))[pP][+-]?\\d+'
    ieeefloat = '0((h|H)[0-9a-fA-F]{4}|(f|F)[0-9a-fA-F]{8}|(d|D)[0-9a-fA-F]{16})'
    tokens = {'root': [include('whitespace'), include('comments'), (string, String), ('@' + identifier + ':?', Name.Label), (register, Name.Variable.Anonymous), include('keyword'), ('&' + identifier, Name.Variable.Global), ('%' + identifier, Name.Variable), (hexfloat, Number.Hex), ('0[xX][a-fA-F0-9]+', Number.Hex), (ieeefloat, Number.Float), (float, Number.Float), ('\\d+', Number.Integer), ('[=<>{}\\[\\]()*.,:;!]|x\\b', Punctuation)], 'whitespace': [('(\\n|\\s)+', Text)], 'comments': [('/\\*.*?\\*/', Comment.Multiline), ('//.*?\\n', Comment.Singleline)], 'keyword': [('kernarg' + datatypeMod, Keyword.Type), ('\\$(full|base|small|large|default|zero|near)', Keyword), (words(('module', 'extension', 'pragma', 'prog', 'indirect', 'signature', 'decl', 'kernel', 'function', 'enablebreakexceptions', 'enabledetectexceptions', 'maxdynamicgroupsize', 'maxflatgridsize', 'maxflatworkgroupsize', 'requireddim', 'requiredgridsize', 'requiredworkgroupsize', 'requirenopartialworkgroups'), suffix='\\b'), Keyword), (roundingMod, Keyword), (datatypeMod, Keyword), ('_(' + alignQual + '|' + widthQual + ')', Keyword), ('_kernarg', Keyword), ('(nop|imagefence)\\b', Keyword), (words(('cleardetectexcept', 'clock', 'cuid', 'debugtrap', 'dim', 'getdetectexcept', 'groupbaseptr', 'kernargbaseptr', 'laneid', 'maxcuid', 'maxwaveid', 'packetid', 'setdetectexcept', 'waveid', 'workitemflatabsid', 'workitemflatid', 'nullptr', 'abs', 'bitrev', 'currentworkgroupsize', 'currentworkitemflatid', 'fract', 'ncos', 'neg', 'nexp2', 'nlog2', 'nrcp', 'nrsqrt', 'nsin', 'nsqrt', 'gridgroups', 'gridsize', 'not', 'sqrt', 'workgroupid', 'workgroupsize', 'workitemabsid', 'workitemid', 'ceil', 'floor', 'rint', 'trunc', 'add', 'bitmask', 'borrow', 'carry', 'copysign', 'div', 'rem', 'sub', 'shl', 'shr', 'and', 'or', 'xor', 'unpackhi', 'unpacklo', 'max', 'min', 'fma', 'mad', 'bitextract', 'bitselect', 'shuffle', 'cmov', 'bitalign', 'bytealign', 'lerp', 'nfma', 'mul', 'mulhi', 'mul24hi', 'mul24', 'mad24', 'mad24hi', 'bitinsert', 'combine', 'expand', 'lda', 'mov', 'pack', 'unpack', 'packcvt', 'unpackcvt', 'sad', 'sementp', 'ftos', 'stof', 'cmp', 'ld', 'st', '_eq', '_ne', '_lt', '_le', '_gt', '_ge', '_equ', '_neu', '_ltu', '_leu', '_gtu', '_geu', '_num', '_nan', '_seq', '_sne', '_slt', '_sle', '_sgt', '_sge', '_snum', '_snan', '_sequ', '_sneu', '_sltu', '_sleu', '_sgtu', '_sgeu', 'atomic', '_ld', '_st', '_cas', '_add', '_and', '_exch', '_max', '_min', '_or', '_sub', '_wrapdec', '_wrapinc', '_xor', 'ret', 'cvt', '_readonly', '_kernarg', '_global', 'br', 'cbr', 'sbr', '_scacq', '_screl', '_scar', '_rlx', '_wave', '_wg', '_agent', '_system', 'ldimage', 'stimage', '_v2', '_v3', '_v4', '_1d', '_2d', '_3d', '_1da', '_2da', '_1db', '_2ddepth', '_2dadepth', '_width', '_height', '_depth', '_array', '_channelorder', '_channeltype', 'querysampler', '_coord', '_filter', '_addressing', 'barrier', 'wavebarrier', 'initfbar', 'joinfbar', 'waitfbar', 'arrivefbar', 'leavefbar', 'releasefbar', 'ldf', 'activelaneid', 'activelanecount', 'activelanemask', 'activelanepermute', 'call', 'scall', 'icall', 'alloca', 'packetcompletionsig', 'addqueuewriteindex', 'casqueuewriteindex', 'ldqueuereadindex', 'stqueuereadindex', 'readonly', 'global', 'private', 'group', 'spill', 'arg', '_upi', '_downi', '_zeroi', '_neari', '_upi_sat', '_downi_sat', '_zeroi_sat', '_neari_sat', '_supi', '_sdowni', '_szeroi', '_sneari', '_supi_sat', '_sdowni_sat', '_szeroi_sat', '_sneari_sat', '_pp', '_ps', '_sp', '_ss', '_s', '_p', '_pp_sat', '_ps_sat', '_sp_sat', '_ss_sat', '_s_sat', '_p_sat')), Keyword), ('i[1-9]\\d*', Keyword)]}