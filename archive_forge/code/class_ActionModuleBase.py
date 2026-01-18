from __future__ import absolute_import, division, print_function
import abc
import copy
import traceback
from ansible import constants as C
from ansible.errors import AnsibleError
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleFallbackNotFound, SEQUENCETYPE, remove_values
from ansible.module_utils.common._collections_compat import (
from ansible.module_utils.common.parameters import (
from ansible.module_utils.common.validation import (
from ansible.module_utils.common.text.formatters import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.plugins.action import ActionBase
@six.add_metaclass(abc.ABCMeta)
class ActionModuleBase(ActionBase):

    @abc.abstractmethod
    def setup_module(self):
        """Return pair (ArgumentSpec, kwargs)."""
        pass

    @abc.abstractmethod
    def run_module(self, module):
        """Run module code"""
        module.fail_json(msg='Not implemented.')

    def run(self, tmp=None, task_vars=None):
        if task_vars is None:
            task_vars = dict()
        result = super(ActionModuleBase, self).run(tmp, task_vars)
        del tmp
        try:
            argument_spec, kwargs = self.setup_module()
            module = argument_spec.create_ansible_module_helper(AnsibleActionModule, (self,), **kwargs)
            self.run_module(module)
            raise AnsibleError('Internal error: action module did not call module.exit_json()')
        except _ModuleExitException as mee:
            result.update(mee.result)
            return result
        except Exception as dummy:
            result['failed'] = True
            result['msg'] = 'MODULE FAILURE'
            result['exception'] = traceback.format_exc()
            return result