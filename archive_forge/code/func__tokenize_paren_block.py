import re
import os_ken.exception
from os_ken.lib.ofctl_utils import str_to_int
from os_ken.ofproto import nicira_ext
def _tokenize_paren_block(string, pos):
    paren_re = re.compile('[()]')
    paren_level = string[:pos].count('(') - string[:pos].count(')')
    while paren_level > 0:
        m = paren_re.search(string[pos:])
        if m.group(0) == '(':
            paren_level += 1
        else:
            paren_level -= 1
        pos += m.end(0)
    return (string[:pos], string[pos:])