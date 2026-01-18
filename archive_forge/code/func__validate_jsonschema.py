from __future__ import absolute_import, division, print_function
import json
from ansible.errors import AnsibleError
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import string_types
from ansible.utils.display import Display
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import to_list
from ansible_collections.ansible.utils.plugins.plugin_utils.base.validate import ValidateBase
def _validate_jsonschema(self):
    error_messages = None
    draft = self._get_sub_plugin_options('draft')
    check_format = self._get_sub_plugin_options('check_format')
    error_messages = []
    for criteria in self._criteria:
        format_checker = None
        validator_class = None
        if draft is not None:
            try:
                validator_class = self._JSONSCHEMA_DRAFTS[draft]['validator']
            except KeyError:
                display.warning('No validator available for "{draft}", falling back to autodetection. A newer version of jsonschema might support this draft.'.format(draft=draft))
        if validator_class is None:
            validator_class = jsonschema.validators.validator_for(criteria)
        if check_format:
            try:
                format_checker = validator_class.FORMAT_CHECKER
            except AttributeError:
                for draft, draft_config in self._JSONSCHEMA_DRAFTS.items():
                    if validator_class == draft_config['validator']:
                        display.vvv('Using format_checker for {draft} validator'.format(draft=draft))
                        format_checker = draft_config['format_checker']
                        break
                else:
                    display.warning('jsonschema format checks not available')
        validator = validator_class(criteria, format_checker=format_checker)
        validation_errors = sorted(validator.iter_errors(self._data), key=lambda e: e.path)
        if validation_errors:
            if 'errors' not in self._result:
                self._result['errors'] = []
            for validation_error in validation_errors:
                if isinstance(validation_error, jsonschema.ValidationError):
                    error = {'message': validation_error.message, 'data_path': to_path(validation_error.absolute_path), 'json_path': json_path(validation_error.absolute_path), 'schema_path': to_path(validation_error.relative_schema_path), 'relative_schema': validation_error.schema, 'expected': validation_error.validator_value, 'validator': validation_error.validator, 'found': validation_error.instance}
                    self._result['errors'].append(error)
                    error_message = "At '{schema_path}' {message}. ".format(schema_path=error['schema_path'], message=error['message'])
                    error_messages.append(error_message)
    if error_messages:
        if 'msg' not in self._result:
            self._result['msg'] = '\n'.join(error_messages)
        else:
            self._result['msg'] += '\n'.join(error_messages)