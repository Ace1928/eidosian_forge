from __future__ import absolute_import, division, print_function
import os
from importlib import import_module
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native, to_text
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import to_list
def _set_sub_plugin_options(self, doc):
    params = {}
    try:
        argspec_obj = yaml.load(doc, SafeLoader)
    except Exception as exc:
        raise AnsibleError("Error '{err}' while reading validate plugin {engine} documentation: '{argspec}'".format(err=to_text(exc, errors='surrogate_or_strict'), engine=self._engine, argspec=doc))
    options = argspec_obj.get('options', {})
    if not options:
        return None
    for option_name, option_value in iteritems(options):
        option_var_name_list = option_value.get('vars', [])
        option_env_name_list = option_value.get('env', [])
        if option_name in self._kwargs:
            params[option_name] = self._kwargs[option_name]
            continue
        if option_var_name_list and option_name not in params:
            for var_name_entry in to_list(option_var_name_list):
                if not isinstance(var_name_entry, dict):
                    raise AnsibleError("invalid type '{var_name_type}' for the value of '{var_name_entry}' option, should to be type dict".format(var_name_type=type(var_name_entry), var_name_entry=var_name_entry))
                var_name = var_name_entry.get('name')
                if var_name and var_name in self._plugin_vars:
                    params[option_name] = self._plugin_vars[var_name]
                    break
        if option_env_name_list and option_name not in params:
            for env_name_entry in to_list(option_env_name_list):
                if not isinstance(env_name_entry, dict):
                    raise AnsibleError("invalid type '{env_name_entry_type}' for the value of '{env_name_entry}' option, should to be type dict".format(env_name_entry_type=type(env_name_entry), env_name_entry=env_name_entry))
                env_name = env_name_entry.get('name')
                if env_name in os.environ:
                    params[option_name] = os.environ[env_name]
                    break
    valid, argspec_result, updated_params = check_argspec(yaml.dump(argspec_obj), self._engine, **params)
    if not valid:
        raise AnsibleError('{argspec_result} with errors: {argspec_errors}'.format(argspec_result=argspec_result.get('msg'), argspec_errors=argspec_result.get('errors')))
    if updated_params:
        self._sub_plugin_options = updated_params