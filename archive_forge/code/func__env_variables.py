import breezy
from breezy import config, i18n, osutils, registry
from another side removing lines.
def _env_variables(topic):
    import textwrap
    ret = ['Environment Variables\n\nSee brz help configuration for more details.\n\n']
    max_key_len = max([len(k[0]) for k in known_env_variables])
    desc_len = 80 - max_key_len - 2
    ret.append('=' * max_key_len + ' ' + '=' * desc_len + '\n')
    for k, desc in known_env_variables:
        ret.append(k + (max_key_len + 1 - len(k)) * ' ')
        ret.append('\n'.join(textwrap.wrap(desc, width=desc_len, subsequent_indent=' ' * (max_key_len + 1))))
        ret.append('\n')
    ret += '=' * max_key_len + ' ' + '=' * desc_len + '\n'
    return ''.join(ret)