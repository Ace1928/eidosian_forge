from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.basic import env_fallback
class MissingDeviceException(CommitException):

    def __init__(self, device_name):
        super(MissingDeviceException, self).__init__('Device missing', 'Device ' + repr(device_name) + ' does not exist')