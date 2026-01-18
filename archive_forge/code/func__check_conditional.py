from __future__ import (absolute_import, division, print_function)
import typing as t
from ansible.errors import AnsibleError, AnsibleUndefinedVariable, AnsibleTemplateError
from ansible.module_utils.common.text.converters import to_native
from ansible.playbook.attribute import FieldAttribute
from ansible.template import Templar
from ansible.utils.display import Display
def _check_conditional(self, conditional: str, templar: Templar, all_vars: dict[str, t.Any]) -> bool:
    original = conditional
    templar.available_variables = all_vars
    try:
        if templar.is_template(conditional):
            display.warning('conditional statements should not include jinja2 templating delimiters such as {{ }} or {%% %%}. Found: %s' % conditional)
            conditional = templar.template(conditional)
            if isinstance(conditional, bool):
                return conditional
            elif conditional == '':
                return False
        if hasattr(conditional, '__UNSAFE__'):
            raise AnsibleTemplateError('Conditional is marked as unsafe, and cannot be evaluated.')
        return templar.template('{%% if %s %%} True {%% else %%} False {%% endif %%}' % conditional).strip() == 'True'
    except AnsibleUndefinedVariable as e:
        raise AnsibleUndefinedVariable('error while evaluating conditional (%s): %s' % (original, e))