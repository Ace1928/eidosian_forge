from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import dict_merge
class AnsibleArgSpecValidator:

    def __init__(self, data, schema, schema_format='doc', schema_conditionals=None, name=None, other_args=None):
        """Validate some data against a schema
        :param data: The data to validate
        :type data: dict
        :param schema: A schema in ansible argspec format
        :type schema: dict
        :param schema_format: 'doc' (ansible docstring) or 'argspec' (ansible argspec)
        :type schema: str if doc, dict if argspec
        :param schema_conditionals: A dict of schema conditionals, ie required_if
        :type schema_conditionals: dict
        :param name: the name of the plugin calling this class, used in error messages
        :type name: str
        :param other_args: Other valid kv pairs for the argspec, eg no_log, bypass_checks
        :type other_args: dict

        note:
        - the schema conditionals can be root conditionals or deeply nested conditionals
          these get dict_merged into the argspec from the docstring, since the docstring cannot
          contain them.
        """
        self._errors = ''
        self._name = name
        self._other_args = other_args
        self._schema = schema
        self._schema_format = schema_format
        self._schema_conditionals = schema_conditionals
        self._data = data

    def _extract_schema_from_doc(self, doc_obj, temp_schema):
        """Extract the schema from a doc string
        :param doc_obj: The doc as a python obj
        :type doc_obj: dictionary
        :params temp_schema: The dict in which we stuff the schema parts
        :type temp_schema: dict
        """
        options_obj = doc_obj.get('options')
        for okey, ovalue in iteritems(options_obj):
            temp_schema[okey] = {}
            for metakey in list(ovalue):
                if metakey == 'suboptions':
                    temp_schema[okey].update({'options': {}})
                    suboptions_obj = {'options': ovalue['suboptions']}
                    self._extract_schema_from_doc(suboptions_obj, temp_schema[okey]['options'])
                elif metakey in OPTION_METADATA + OPTION_CONDITIONALS:
                    temp_schema[okey].update({metakey: ovalue[metakey]})

    def _convert_doc_to_schema(self):
        """Convert the doc string to an obj, was yaml
        add back other valid conditionals and params
        """
        doc_obj = yaml.load(self._schema, SafeLoader)
        temp_schema = {}
        self._extract_schema_from_doc(doc_obj, temp_schema)
        self._schema = {'argument_spec': temp_schema}

    def _validate(self):
        """Validate the data gainst the schema
        convert doc string in argspec if necessary

        :return valid: if the data passed
        :rtype valid: bool
        :return errors: errors reported during validation
        :rtype errors: str
        :return params: The original data updated with defaults
        :rtype params: dict
        """
        if self._schema_format == 'doc':
            self._convert_doc_to_schema()
        if self._schema_conditionals is not None:
            self._schema = dict_merge(self._schema, self._schema_conditionals)
        if self._other_args is not None:
            self._schema = dict_merge(self._schema, self._other_args)
        invalid_keys = [k for k in self._schema.keys() if k not in VALID_ANSIBLEMODULE_ARGS]
        if invalid_keys:
            valid = False
            errors = 'Invalid schema. Invalid keys found: {ikeys}'.format(ikeys=','.join(invalid_keys))
            updated_data = {}
        else:
            mm = MonkeyModule(data=self._data, schema=self._schema, name=self._name)
            valid, errors, updated_data = mm.validate()
        return (valid, errors, updated_data)

    def validate(self):
        """The public validate method
        check for future argspec validation
        that is coming in 2.11, change the check according above
        """
        if HAS_ANSIBLE_ARG_SPEC_VALIDATOR:
            if self._schema_format == 'doc':
                self._convert_doc_to_schema()
            if self._schema_conditionals is not None:
                self._schema = dict_merge(self._schema, self._schema_conditionals)
            invalid_keys = [k for k in self._schema.keys() if k not in VALID_ANSIBLEMODULE_ARGS]
            if invalid_keys:
                valid = False
                errors = ['Invalid schema. Invalid keys found: {ikeys}'.format(ikeys=','.join(invalid_keys))]
                updated_data = {}
                return (valid, errors, updated_data)
            else:
                validator = ArgumentSpecValidator(**self._schema)
                result = validator.validate(self._data)
                valid = not bool(result.error_messages)
                return (valid, result.error_messages, result.validated_parameters)
        else:
            return self._validate()