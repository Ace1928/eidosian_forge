from __future__ import (absolute_import, division, print_function)
import hashlib
import os
import string
from collections.abc import Mapping
from ansible.errors import AnsibleError, AnsibleParserError
from ansible.inventory.group import to_safe_group_name as original_safe
from ansible.parsing.utils.addresses import parse_address
from ansible.plugins import AnsiblePlugin
from ansible.plugins.cache import CachePluginAdjudicator as CacheObject
from ansible.module_utils.common.text.converters import to_bytes, to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible.template import Templar
from ansible.utils.display import Display
from ansible.utils.vars import combine_vars, load_extra_vars
class Constructable(object):

    def _compose(self, template, variables, disable_lookups=True):
        """ helper method for plugins to compose variables for Ansible based on jinja2 expression and inventory vars"""
        t = self.templar
        try:
            use_extra = self.get_option('use_extra_vars')
        except Exception:
            use_extra = False
        if use_extra:
            t.available_variables = combine_vars(variables, self._vars)
        else:
            t.available_variables = variables
        return t.template('%s%s%s' % (t.environment.variable_start_string, template, t.environment.variable_end_string), disable_lookups=disable_lookups)

    def _set_composite_vars(self, compose, variables, host, strict=False):
        """ loops over compose entries to create vars for hosts """
        if compose and isinstance(compose, dict):
            for varname in compose:
                try:
                    composite = self._compose(compose[varname], variables)
                except Exception as e:
                    if strict:
                        raise AnsibleError('Could not set %s for host %s: %s' % (varname, host, to_native(e)))
                    continue
                self.inventory.set_variable(host, varname, composite)

    def _add_host_to_composed_groups(self, groups, variables, host, strict=False, fetch_hostvars=True):
        """ helper to create complex groups for plugins based on jinja2 conditionals, hosts that meet the conditional are added to group"""
        if groups and isinstance(groups, dict):
            if fetch_hostvars:
                variables = combine_vars(variables, self.inventory.get_host(host).get_vars())
            self.templar.available_variables = variables
            for group_name in groups:
                conditional = '{%% if %s %%} True {%% else %%} False {%% endif %%}' % groups[group_name]
                group_name = self._sanitize_group_name(group_name)
                try:
                    result = boolean(self.templar.template(conditional))
                except Exception as e:
                    if strict:
                        raise AnsibleParserError('Could not add host %s to group %s: %s' % (host, group_name, to_native(e)))
                    continue
                if result:
                    group_name = self.inventory.add_group(group_name)
                    self.inventory.add_child(group_name, host)

    def _add_host_to_keyed_groups(self, keys, variables, host, strict=False, fetch_hostvars=True):
        """ helper to create groups for plugins based on variable values and add the corresponding hosts to it"""
        if keys and isinstance(keys, list):
            for keyed in keys:
                if keyed and isinstance(keyed, dict):
                    if fetch_hostvars:
                        variables = combine_vars(variables, self.inventory.get_host(host).get_vars())
                    try:
                        key = self._compose(keyed.get('key'), variables)
                    except Exception as e:
                        if strict:
                            raise AnsibleParserError('Could not generate group for host %s from %s entry: %s' % (host, keyed.get('key'), to_native(e)))
                        continue
                    default_value_name = keyed.get('default_value', None)
                    trailing_separator = keyed.get('trailing_separator')
                    if trailing_separator is not None and default_value_name is not None:
                        raise AnsibleParserError('parameters are mutually exclusive for keyed groups: default_value|trailing_separator')
                    if key or (key == '' and default_value_name is not None):
                        prefix = keyed.get('prefix', '')
                        sep = keyed.get('separator', '_')
                        raw_parent_name = keyed.get('parent_group', None)
                        if raw_parent_name:
                            try:
                                raw_parent_name = self.templar.template(raw_parent_name)
                            except AnsibleError as e:
                                if strict:
                                    raise AnsibleParserError('Could not generate parent group %s for group %s: %s' % (raw_parent_name, key, to_native(e)))
                                continue
                        new_raw_group_names = []
                        if isinstance(key, string_types):
                            if key == '' and default_value_name is not None:
                                new_raw_group_names.append(default_value_name)
                            else:
                                new_raw_group_names.append(key)
                        elif isinstance(key, list):
                            for name in key:
                                if name == '' and default_value_name is not None:
                                    new_raw_group_names.append(default_value_name)
                                else:
                                    new_raw_group_names.append(name)
                        elif isinstance(key, Mapping):
                            for gname, gval in key.items():
                                bare_name = '%s%s%s' % (gname, sep, gval)
                                if gval == '':
                                    if default_value_name is not None:
                                        bare_name = '%s%s%s' % (gname, sep, default_value_name)
                                    elif trailing_separator is False:
                                        bare_name = gname
                                new_raw_group_names.append(bare_name)
                        else:
                            raise AnsibleParserError('Invalid group name format, expected a string or a list of them or dictionary, got: %s' % type(key))
                        for bare_name in new_raw_group_names:
                            if prefix == '' and self.get_option('leading_separator') is False:
                                sep = ''
                            gname = self._sanitize_group_name('%s%s%s' % (prefix, sep, bare_name))
                            result_gname = self.inventory.add_group(gname)
                            self.inventory.add_host(host, result_gname)
                            if raw_parent_name:
                                parent_name = self._sanitize_group_name(raw_parent_name)
                                self.inventory.add_group(parent_name)
                                self.inventory.add_child(parent_name, result_gname)
                    elif strict and key not in ([], {}):
                        raise AnsibleParserError('No key or key resulted empty for %s in host %s, invalid entry' % (keyed.get('key'), host))
                else:
                    raise AnsibleParserError('Invalid keyed group entry, it must be a dictionary: %s ' % keyed)