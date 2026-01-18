from __future__ import (absolute_import, division, print_function)
import subprocess
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.common.text.converters import to_text
class LookupModule(LookupBase):

    def run(self, terms, variables, **kwargs):
        ret = []
        for term in terms:
            p = subprocess.Popen(term, cwd=self._loader.get_basedir(), shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            stdout, stderr = p.communicate()
            if p.returncode == 0:
                ret.extend([to_text(l) for l in stdout.splitlines()])
            else:
                raise AnsibleError('lookup_plugin.lines(%s) returned %d' % (term, p.returncode))
        return ret