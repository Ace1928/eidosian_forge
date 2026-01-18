from __future__ import absolute_import, division, print_function
import traceback
import re
import json
from itertools import chain
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils._text import to_native
from ansible.module_utils.common.collections import is_iterable
from ansible.module_utils.basic import AnsibleModule, missing_required_lib, _load_params
from ansible.module_utils.urls import open_url
class NetboxAnsibleModule(AnsibleModule):
    """
    Creating this due to needing to override some functionality to provide required_together, required_if
    and will be able to override more in the future.
    This is due to the NetBox modules having the module arguments within a key in the argument spec, using suboptions rather than
    having all module arguments within the regular argument spec.

    Didn't want to change that functionality of the NetBox modules as its disruptive and we're required to send a specific payload
    to the NetBox API
    """

    def __init__(self, argument_spec, bypass_checks=False, no_log=False, mutually_exclusive=None, required_together=None, required_one_of=None, add_file_common_args=False, supports_check_mode=False, required_if=None, required_by=None):
        super().__init__(argument_spec, bypass_checks=False, no_log=False, mutually_exclusive=None, required_together=None, required_one_of=None, add_file_common_args=False, supports_check_mode=supports_check_mode, required_if=None)
        params = _load_params()
        if mutually_exclusive:
            self._check_mutually_exclusive(mutually_exclusive, param=params)
        if required_together:
            self._check_required_together(required_together, param=params)
        if required_one_of:
            self._check_required_one_of(required_one_of, param=params)
        if required_if:
            self._check_required_if(required_if, param=params)

    def _check_mutually_exclusive(self, spec, param=None):
        if param is None:
            param = self.params
        try:
            self.check_mutually_exclusive(spec, param)
        except TypeError as e:
            msg = to_native(e)
            if self._options_context:
                msg += ' found in %s' % ' -> '.join(self._options_context)
            self.fail_json(msg=msg)

    def check_mutually_exclusive(self, terms, module_parameters):
        """Check mutually exclusive terms against argument parameters
        Accepts a single list or list of lists that are groups of terms that should be
        mutually exclusive with one another
        :arg terms: List of mutually exclusive module parameters
        :arg module_parameters: Dictionary of module parameters
        :returns: Empty list or raises TypeError if the check fails.
        """
        results = []
        if terms is None:
            return results
        for check in terms:
            count = self.count_terms(check, module_parameters['data'])
            if count > 1:
                results.append(check)
        if results:
            full_list = ['|'.join(check) for check in results]
            msg = 'parameters are mutually exclusive: %s' % ', '.join(full_list)
            raise TypeError(to_native(msg))
        return results

    def _check_required_if(self, spec, param=None):
        """ensure that parameters which conditionally required are present"""
        if spec is None:
            return
        if param is None:
            param = self.params
        try:
            self.check_required_if(spec, param)
        except TypeError as e:
            msg = to_native(e)
            if self._options_context:
                msg += ' found in %s' % ' -> '.join(self._options_context)
            self.fail_json(msg=msg)

    def check_required_if(self, requirements, module_parameters):
        results = []
        if requirements is None:
            return results
        for req in requirements:
            missing = {}
            missing['missing'] = []
            max_missing_count = 0
            is_one_of = False
            if len(req) == 4:
                key, val, requirements, is_one_of = req
            else:
                key, val, requirements = req
            if is_one_of:
                max_missing_count = len(requirements)
                missing['requires'] = 'any'
            else:
                missing['requires'] = 'all'
            if key in module_parameters and module_parameters[key] == val:
                for check in requirements:
                    count = self.count_terms(check, module_parameters['data'])
                    if count == 0:
                        missing['missing'].append(check)
            if len(missing['missing']) and len(missing['missing']) >= max_missing_count:
                missing['parameter'] = key
                missing['value'] = val
                missing['requirements'] = requirements
                results.append(missing)
        if results:
            for missing in results:
                msg = '%s is %s but %s of the following are missing: %s' % (missing['parameter'], missing['value'], missing['requires'], ', '.join(missing['missing']))
                raise TypeError(to_native(msg))
        return results

    def _check_required_one_of(self, spec, param=None):
        if spec is None:
            return
        if param is None:
            param = self.params
        try:
            self.check_required_one_of(spec, param)
        except TypeError as e:
            msg = to_native(e)
            if self._options_context:
                msg += ' found in %s' % ' -> '.join(self._options_context)
            self.fail_json(msg=msg)

    def check_required_one_of(self, terms, module_parameters):
        """Check each list of terms to ensure at least one exists in the given module
        parameters
        Accepts a list of lists or tuples
        :arg terms: List of lists of terms to check. For each list of terms, at
            least one is required.
        :arg module_parameters: Dictionary of module parameters
        :returns: Empty list or raises TypeError if the check fails.
        """
        results = []
        if terms is None:
            return results
        for term in terms:
            count = self.count_terms(term, module_parameters['data'])
            if count == 0:
                results.append(term)
        if results:
            for term in results:
                msg = 'one of the following is required: %s' % ', '.join(term)
                raise TypeError(to_native(msg))
        return results

    def _check_required_together(self, spec, param=None):
        if spec is None:
            return
        if param is None:
            param = self.params
        try:
            self.check_required_together(spec, param)
        except TypeError as e:
            msg = to_native(e)
            if self._options_context:
                msg += ' found in %s' % ' -> '.join(self._options_context)
            self.fail_json(msg=msg)

    def check_required_together(self, terms, module_parameters):
        """Check each list of terms to ensure every parameter in each list exists
        in the given module parameters
        Accepts a list of lists or tuples
        :arg terms: List of lists of terms to check. Each list should include
            parameters that are all required when at least one is specified
            in the module_parameters.
        :arg module_parameters: Dictionary of module parameters
        :returns: Empty list or raises TypeError if the check fails.
        """
        results = []
        if terms is None:
            return results
        for term in terms:
            counts = [self.count_terms(field, module_parameters['data']) for field in term]
            non_zero = [c for c in counts if c > 0]
            if len(non_zero) > 0:
                if 0 in counts:
                    results.append(term)
        if results:
            for term in results:
                msg = 'parameters are required together: %s' % ', '.join(term)
                raise TypeError(to_native(msg))
        return results

    def count_terms(self, terms, module_parameters):
        """Count the number of occurrences of a key in a given dictionary
        :arg terms: String or iterable of values to check
        :arg module_parameters: Dictionary of module parameters
        :returns: An integer that is the number of occurrences of the terms values
        in the provided dictionary.
        """
        if not is_iterable(terms):
            terms = [terms]
        return len(set(terms).intersection(module_parameters))