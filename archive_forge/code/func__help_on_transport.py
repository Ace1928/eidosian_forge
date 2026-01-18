import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def _help_on_transport(name):
    import textwrap
    from breezy.transport import transport_list_registry

    def add_string(proto, help, maxl, prefix_width=20):
        help_lines = textwrap.wrap(help, maxl - prefix_width, break_long_words=False)
        line_with_indent = '\n' + ' ' * prefix_width
        help_text = line_with_indent.join(help_lines)
        return '%-20s%s\n' % (proto, help_text)

    def key_func(a):
        return a[:a.rfind('://')]
    protl = []
    decl = []
    protos = transport_list_registry.keys()
    protos.sort(key=key_func)
    for proto in protos:
        shorthelp = transport_list_registry.get_help(proto)
        if not shorthelp:
            continue
        if proto.endswith('://'):
            protl.append(add_string(proto, shorthelp, 79))
        else:
            decl.append(add_string(proto, shorthelp, 79))
    out = 'URL Identifiers\n\n' + 'Supported URL prefixes::\n\n  ' + '  '.join(protl)
    if len(decl):
        out += '\nSupported modifiers::\n\n  ' + '  '.join(decl)
    out += "\nBreezy supports all of the standard parts within the URL::\n\n  <protocol>://[user[:password]@]host[:port]/[path]\n\nallowing URLs such as::\n\n  http://brzuser:BadPass@brz.example.com:8080/brz/trunk\n\nFor brz+ssh:// and sftp:// URLs, Breezy also supports paths that begin\nwith '~' as meaning that the rest of the path should be interpreted\nrelative to the remote user's home directory.  For example if the user\n``remote`` has a  home directory of ``/home/remote`` on the server\nshell.example.com, then::\n\n  brz+ssh://remote@shell.example.com/~/myproject/trunk\n\nwould refer to ``/home/remote/myproject/trunk``.\n\nMany commands that accept URLs also accept location aliases too.\nSee :doc:`location-alias-help` and :doc:`url-special-chars-help`.\n"
    return out