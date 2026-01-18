from __future__ import absolute_import, division, print_function
import copy
import os
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_text
def _content_of_file_at_path(path):
    """Read file content.

    Try read content of file at specified path.
    :type path:  str
    :param path: Full path to location of file which should be read'ed.
    :rtype:  content
    :return: File content or 'None'
    """
    content = None
    if path and os.path.exists(path):
        with open(path, mode='rt') as opened_file:
            b_content = opened_file.read()
            try:
                content = to_text(b_content, errors='surrogate_or_strict')
            except UnicodeError:
                pass
    return content