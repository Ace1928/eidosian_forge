from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems
from ansible_collections.ansible.utils.plugins.module_utils.common.utils import dict_merge
class MonkeyModule(AnsibleModule):
    """A derivative of the AnsibleModule used
    to just validate the data (task.args) against
    the schema(argspec)
    """

    def __init__(self, data, schema, name):
        self._errors = None
        self._valid = True
        self._schema = schema
        self.name = name
        self.params = data

    def fail_json(self, msg):
        """Replace the AnsibleModule fail_json here
        :param msg: The message for the failure
        :type msg: str
        """
        if self.name:
            msg = re.sub('\\(basic\\.pyc?\\)', "'{name}'".format(name=self.name), msg)
        self._valid = False
        self._errors = msg

    def _load_params(self):
        """This replaces the AnsibleModule _load_params
        fn because we already set self.params in init
        """
        pass

    def validate(self):
        """Instantiate the super, validating the schema
        against the data
        :return valid: if the data passed
        :rtype valid: bool
        :return errors: errors reported during validation
        :rtype errors: str
        :return params: The original data updated with defaults
        :rtype params: dict
        """
        super(MonkeyModule, self).__init__(**self._schema)
        return (self._valid, self._errors, self.params)