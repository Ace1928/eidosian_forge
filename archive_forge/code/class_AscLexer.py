import re
from pygments.lexer import RegexLexer, bygroups
from pygments.token import Comment, Generic, Name, Operator, String, Whitespace
class AscLexer(RegexLexer):
    """
    Lexer for ASCII armored files, containing `-----BEGIN/END ...-----` wrapped
    base64 data.

    .. versionadded:: 2.10
    """
    name = 'ASCII armored'
    aliases = ['asc', 'pem']
    filenames = ['*.asc', '*.pem', 'id_dsa', 'id_ecdsa', 'id_ecdsa_sk', 'id_ed25519', 'id_ed25519_sk', 'id_rsa']
    mimetypes = ['application/pgp-keys', 'application/pgp-encrypted', 'application/pgp-signature', 'application/pem-certificate-chain']
    flags = re.MULTILINE
    tokens = {'root': [('\\s+', Whitespace), ('^-----BEGIN [^\\n]+-----$', Generic.Heading, 'data'), ('\\S+', Comment)], 'data': [('\\s+', Whitespace), ('^([^:]+)(:)([ \\t]+)(.*)', bygroups(Name.Attribute, Operator, Whitespace, String)), ('^-----END [^\\n]+-----$', Generic.Heading, 'root'), ('\\S+', String)]}

    def analyse_text(text):
        if re.search('^-----BEGIN [^\\n]+-----\\r?\\n', text):
            return True