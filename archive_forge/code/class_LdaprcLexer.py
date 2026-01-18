import re
from pygments.lexer import RegexLexer, bygroups, default
from pygments.token import Operator, Comment, Keyword, Literal, Name, String, \
class LdaprcLexer(RegexLexer):
    """
    Lexer for OpenLDAP configuration files.

    .. versionadded:: 2.17
    """
    name = 'LDAP configuration file'
    aliases = ['ldapconf', 'ldaprc']
    filenames = ['.ldaprc', 'ldaprc', 'ldap.conf']
    mimetypes = ['text/x-ldapconf']
    url = 'https://www.openldap.org/software//man.cgi?query=ldap.conf&sektion=5&apropos=0&manpath=OpenLDAP+2.4-Release'
    _sasl_keywords = 'SASL_(?:MECH|REALM|AUTHCID|AUTHZID|CBINDING)'
    _tls_keywords = 'TLS_(?:CACERT|CACERTDIR|CERT|ECNAME|KEY|CIPHER_SUITE|PROTOCOL_MIN|RANDFILE|CRLFILE)'
    _literal_keywords = f'(?:URI|SOCKET_BIND_ADDRESSES|{_sasl_keywords}|{_tls_keywords})'
    _boolean_keywords = 'GSSAPI_(?:ALLOW_REMOTE_PRINCIPAL|ENCRYPT|SIGN)|REFERRALS|SASL_NOCANON'
    _integer_keywords = 'KEEPALIVE_(?:IDLE|PROBES|INTERVAL)|NETWORK_TIMEOUT|PORT|SIZELIMIT|TIMELIMIT|TIMEOUT'
    _secprops = 'none|noanonymous|noplain|noactive|nodict|forwardsec|passcred|(?:minssf|maxssf|maxbufsize)=\\d+'
    flags = re.IGNORECASE | re.MULTILINE
    tokens = {'root': [('#.*', Comment.Single), ('\\s+', Whitespace), (f'({_boolean_keywords})(\\s+)(on|true|yes|off|false|no)$', bygroups(Keyword, Whitespace, Keyword.Constant)), (f'({_integer_keywords})(\\s+)(\\d+)', bygroups(Keyword, Whitespace, Number.Integer)), ('(VERSION)(\\s+)(2|3)', bygroups(Keyword, Whitespace, Number.Integer)), ('(DEREF)(\\s+)(never|searching|finding|always)', bygroups(Keyword, Whitespace, Keyword.Constant)), (f'(SASL_SECPROPS)(\\s+)((?:{_secprops})(?:,{_secprops})*)', bygroups(Keyword, Whitespace, Keyword.Constant)), ('(SASL_CBINDING)(\\s+)(none|tls-unique|tls-endpoint)', bygroups(Keyword, Whitespace, Keyword.Constant)), ('(TLS_REQ(?:CERT|SAN))(\\s+)(allow|demand|hard|never|try)', bygroups(Keyword, Whitespace, Keyword.Constant)), ('(TLS_CRLCHECK)(\\s+)(none|peer|all)', bygroups(Keyword, Whitespace, Keyword.Constant)), ('(BASE|BINDDN)(\\s+)(\\S+)$', bygroups(Keyword, Whitespace, Literal)), ('(HOST)(\\s+)([a-z0-9]+)((?::(\\d+))?)', bygroups(Keyword, Whitespace, Literal, Number.Integer)), (f'({_literal_keywords})(\\s+)(\\S+)$', bygroups(Keyword, Whitespace, Literal))]}