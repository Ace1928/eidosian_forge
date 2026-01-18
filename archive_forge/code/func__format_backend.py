from __future__ import (absolute_import, division, print_function)
def _format_backend(vars):
    if 'api_key_file' in vars:
        protocol = 'wss'
    else:
        protocol = 'ws'
    return '{0}://{1}:{2}'.format(protocol, vars['inventory_hostname'], 8081)