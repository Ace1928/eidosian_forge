from __future__ import (absolute_import, division, print_function)
from collections.abc import Mapping
from ansible import constants as C
from ansible.template import Templar, AnsibleUndefined
class HostVarsVars(Mapping):

    def __init__(self, variables, loader):
        self._vars = variables
        self._loader = loader
        self._templar = Templar(variables=self._vars, loader=self._loader)

    def __getitem__(self, var):
        return self._templar.template(self._vars[var], fail_on_undefined=False, static_vars=C.INTERNAL_STATIC_VARS)

    def __contains__(self, var):
        return var in self._vars

    def __iter__(self):
        for var in self._vars.keys():
            yield var

    def __len__(self):
        return len(self._vars.keys())

    def __repr__(self):
        return repr(self._templar.template(self._vars, fail_on_undefined=False, static_vars=C.INTERNAL_STATIC_VARS))