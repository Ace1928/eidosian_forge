from __future__ import absolute_import, division, print_function
import json
import os
import stat
import ansible_collections.community.general.plugins.module_utils.puppet as puppet_utils
from ansible.module_utils.basic import AnsibleModule
def _write_structured_data(basedir, basename, data):
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    file_path = os.path.join(basedir, '{0}.json'.format(basename))
    out_file = os.fdopen(os.open(file_path, os.O_CREAT | os.O_WRONLY, stat.S_IRUSR | stat.S_IWUSR), 'w')
    out_file.write(json.dumps(data).encode('utf8'))
    out_file.close()