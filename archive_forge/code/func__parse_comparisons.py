from __future__ import absolute_import, division, print_function
import re
from time import sleep
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.docker.plugins.module_utils.util import (
from ansible_collections.community.docker.plugins.module_utils._api.utils.utils import parse_repository_tag
def _parse_comparisons(self):
    all_module_options = self._collect_all_module_params()
    comp_aliases = {}
    for option_name, option in self.all_options.items():
        if option.not_an_ansible_option:
            continue
        comp_aliases[option_name] = option_name
        for alias in option.ansible_aliases:
            comp_aliases[alias] = option_name
    if self.module.params['ignore_image']:
        self.all_options['image'].comparison = 'ignore'
    if self.module.params['purge_networks']:
        self.all_options['networks'].comparison = 'strict'
    if self.module.params.get('comparisons'):
        if '*' in self.module.params['comparisons']:
            value = self.module.params['comparisons']['*']
            if value not in ('strict', 'ignore'):
                self.fail("The wildcard can only be used with comparison modes 'strict' and 'ignore'!")
            for option in self.all_options.values():
                if option.name == 'networks':
                    if self.module.params['networks'] is None:
                        continue
                option.comparison = value
        comp_aliases_used = {}
        for key, value in self.module.params['comparisons'].items():
            if key == '*':
                continue
            key_main = comp_aliases.get(key)
            if key_main is None:
                if key_main in all_module_options:
                    self.fail("The module option '%s' cannot be specified in the comparisons dict, since it does not correspond to container's state!" % key)
                if key not in self.all_options or self.all_options[key].not_an_ansible_option:
                    self.fail("Unknown module option '%s' in comparisons dict!" % key)
                key_main = key
            if key_main in comp_aliases_used:
                self.fail("Both '%s' and '%s' (aliases of %s) are specified in comparisons dict!" % (key, comp_aliases_used[key_main], key_main))
            comp_aliases_used[key_main] = key
            if value in ('strict', 'ignore'):
                self.all_options[key_main].comparison = value
            elif value == 'allow_more_present':
                if self.all_options[key_main].comparison_type == 'value':
                    self.fail("Option '%s' is a value and not a set/list/dict, so its comparison cannot be %s" % (key, value))
                self.all_options[key_main].comparison = value
            else:
                self.fail("Unknown comparison mode '%s'!" % value)
    for option in self.all_options.values():
        if option.copy_comparison_from is not None:
            option.comparison = self.all_options[option.copy_comparison_from].comparison
    if self.module.params['ignore_image'] and self.all_options['image'].comparison != 'ignore':
        self.module.warn('The ignore_image option has been overridden by the comparisons option!')
    if self.module.params['purge_networks'] and self.all_options['networks'].comparison != 'strict':
        self.module.warn('The purge_networks option has been overridden by the comparisons option!')