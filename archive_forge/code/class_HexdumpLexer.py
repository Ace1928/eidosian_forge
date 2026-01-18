from pygments.lexer import RegexLexer, bygroups, include
from pygments.token import Text, Name, Number, String, Punctuation
class HexdumpLexer(RegexLexer):
    """
    For typical hex dump output formats by the UNIX and GNU/Linux tools ``hexdump``,
    ``hd``, ``hexcat``, ``od`` and ``xxd``, and the DOS tool ``DEBUG``. For example:

    .. sourcecode:: hexdump

        00000000  7f 45 4c 46 02 01 01 00  00 00 00 00 00 00 00 00  |.ELF............|
        00000010  02 00 3e 00 01 00 00 00  c5 48 40 00 00 00 00 00  |..>......H@.....|

    The specific supported formats are the outputs of:

    * ``hexdump FILE``
    * ``hexdump -C FILE`` -- the `canonical` format used in the example.
    * ``hd FILE`` -- same as ``hexdump -C FILE``.
    * ``hexcat FILE``
    * ``od -t x1z FILE``
    * ``xxd FILE``
    * ``DEBUG.EXE FILE.COM`` and entering ``d`` to the prompt.

    .. versionadded:: 2.1
    """
    name = 'Hexdump'
    aliases = ['hexdump']
    hd = '[0-9A-Ha-h]'
    tokens = {'root': [('\\n', Text), include('offset'), ('(' + hd + '{2})(\\-)(' + hd + '{2})', bygroups(Number.Hex, Punctuation, Number.Hex)), (hd + '{2}', Number.Hex), ('(\\s{2,3})(\\>)(.{16})(\\<)$', bygroups(Text, Punctuation, String, Punctuation), 'bracket-strings'), ('(\\s{2,3})(\\|)(.{16})(\\|)$', bygroups(Text, Punctuation, String, Punctuation), 'piped-strings'), ('(\\s{2,3})(\\>)(.{1,15})(\\<)$', bygroups(Text, Punctuation, String, Punctuation)), ('(\\s{2,3})(\\|)(.{1,15})(\\|)$', bygroups(Text, Punctuation, String, Punctuation)), ('(\\s{2,3})(.{1,15})$', bygroups(Text, String)), ('(\\s{2,3})(.{16}|.{20})$', bygroups(Text, String), 'nonpiped-strings'), ('\\s', Text), ('^\\*', Punctuation)], 'offset': [('^(' + hd + '+)(:)', bygroups(Name.Label, Punctuation), 'offset-mode'), ('^' + hd + '+', Name.Label)], 'offset-mode': [('\\s', Text, '#pop'), (hd + '+', Name.Label), (':', Punctuation)], 'piped-strings': [('\\n', Text), include('offset'), (hd + '{2}', Number.Hex), ('(\\s{2,3})(\\|)(.{1,16})(\\|)$', bygroups(Text, Punctuation, String, Punctuation)), ('\\s', Text), ('^\\*', Punctuation)], 'bracket-strings': [('\\n', Text), include('offset'), (hd + '{2}', Number.Hex), ('(\\s{2,3})(\\>)(.{1,16})(\\<)$', bygroups(Text, Punctuation, String, Punctuation)), ('\\s', Text), ('^\\*', Punctuation)], 'nonpiped-strings': [('\\n', Text), include('offset'), ('(' + hd + '{2})(\\-)(' + hd + '{2})', bygroups(Number.Hex, Punctuation, Number.Hex)), (hd + '{2}', Number.Hex), ('(\\s{19,})(.{1,20}?)$', bygroups(Text, String)), ('(\\s{2,3})(.{1,20})$', bygroups(Text, String)), ('\\s', Text), ('^\\*', Punctuation)]}