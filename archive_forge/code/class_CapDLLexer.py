from pygments.lexer import RegexLexer, include, words
from pygments.token import Text, Comment, Operator, Keyword, Name, String, \
class CapDLLexer(RegexLexer):
    """
    Basic lexer for
    `CapDL <https://ssrg.nicta.com.au/publications/nictaabstracts/Kuz_KLW_10.abstract.pml>`_.

    The source of the primary tool that reads such specifications is available
    at https://github.com/seL4/capdl/tree/master/capDL-tool. Note that this
    lexer only supports a subset of the grammar. For example, identifiers can
    shadow type names, but these instances are currently incorrectly
    highlighted as types. Supporting this would need a stateful lexer that is
    considered unnecessarily complex for now.

    .. versionadded:: 2.2
    """
    name = 'CapDL'
    aliases = ['capdl']
    filenames = ['*.cdl']
    tokens = {'root': [('^\\s*#.*\\n', Comment.Preproc), ('\\s+', Text), ('/\\*(.|\\n)*?\\*/', Comment), ('(//|--).*\\n', Comment), ('[<>\\[(){},:;=\\]]', Punctuation), ('\\.\\.', Punctuation), (words(('arch', 'arm11', 'caps', 'child_of', 'ia32', 'irq', 'maps', 'objects'), suffix='\\b'), Keyword), (words(('aep', 'asid_pool', 'cnode', 'ep', 'frame', 'io_device', 'io_ports', 'io_pt', 'notification', 'pd', 'pt', 'tcb', 'ut', 'vcpu'), suffix='\\b'), Keyword.Type), (words(('asid', 'addr', 'badge', 'cached', 'dom', 'domainID', 'elf', 'fault_ep', 'G', 'guard', 'guard_size', 'init', 'ip', 'prio', 'sp', 'R', 'RG', 'RX', 'RW', 'RWG', 'RWX', 'W', 'WG', 'WX', 'level', 'masked', 'master_reply', 'paddr', 'ports', 'reply', 'uncached'), suffix='\\b'), Keyword.Reserved), ('0[xX][\\da-fA-F]+', Number.Hex), ('\\d+(\\.\\d+)?(k|M)?', Number), (words(('bits',), suffix='\\b'), Number), (words(('cspace', 'vspace', 'reply_slot', 'caller_slot', 'ipc_buffer_slot'), suffix='\\b'), Number), ('[a-zA-Z_][-@\\.\\w]*', Name)]}