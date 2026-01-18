from uc_micro.categories import Cc, Cf, P, Z
from uc_micro.properties import Any
def _re_src_path(opts):
    src_path = '(?:' + '[/?#]' + '(?:' + '(?!' + SRC_ZCC + '|' + TEXT_SEPARATORS + '|[()[\\]{}.,"\'?!\\-;]).|' + '\\[(?:(?!' + SRC_ZCC + '|\\]).)*\\]|' + '\\((?:(?!' + SRC_ZCC + '|[)]).)*\\)|' + '\\{(?:(?!' + SRC_ZCC + '|[}]).)*\\}|' + '\\"(?:(?!' + SRC_ZCC + '|["]).)+\\"|' + "\\'(?:(?!" + SRC_ZCC + "|[']).)+\\'|" + "\\'(?=" + SRC_PSEUDO_LETTER + '|[-])|' + '\\.{2,}[a-zA-Z0-9%/&]|' + '\\.(?!' + SRC_ZCC + '|[.]|$)|' + ('\\-(?!--(?:[^-]|$))(?:-*)|' if opts.get('---') else '\\-+|') + ',(?!' + SRC_ZCC + '|$)|' + ';(?!' + SRC_ZCC + '|$)|' + '\\!+(?!' + SRC_ZCC + '|[!]|$)|' + '\\?(?!' + SRC_ZCC + '|[?]|$)' + ')+' + '|\\/' + ')?'
    return src_path