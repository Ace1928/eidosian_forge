from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class MissingActionParametersException(RunnableException):

    def __init__(self, required_parameters):
        super(MissingActionParametersException, self).__init__('Action parameters missing', 'Required parameters are: ' + repr(required_parameters))