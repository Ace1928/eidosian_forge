from __future__ import absolute_import, division, print_function
import traceback
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils._text import to_native
import ansible_collections.netapp.ontap.plugins.module_utils.netapp as netapp_utils
@classmethod
def _format_escaped_data(cls, datastring):
    """ replace helper escape sequences """
    formatted_string = datastring.replace('------', '---').replace('---', '\n').replace('###', '    ').strip()
    retval_string = ''
    for line in formatted_string.split('\n'):
        stripped_line = line.strip()
        if len(stripped_line) > 1:
            retval_string += stripped_line + '\n'
    return retval_string