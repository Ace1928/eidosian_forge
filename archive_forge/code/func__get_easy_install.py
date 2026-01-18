from __future__ import absolute_import, division, print_function
import os
import os.path
import tempfile
from ansible.module_utils.basic import AnsibleModule
def _get_easy_install(module, env=None, executable=None):
    candidate_easy_inst_basenames = ['easy_install']
    easy_install = None
    if executable is not None:
        if os.path.isabs(executable):
            easy_install = executable
        else:
            candidate_easy_inst_basenames.insert(0, executable)
    if easy_install is None:
        if env is None:
            opt_dirs = []
        else:
            opt_dirs = ['%s/bin' % env]
        for basename in candidate_easy_inst_basenames:
            easy_install = module.get_bin_path(basename, False, opt_dirs)
            if easy_install is not None:
                break
    if easy_install is None:
        basename = candidate_easy_inst_basenames[0]
        easy_install = module.get_bin_path(basename, True, opt_dirs)
    return easy_install