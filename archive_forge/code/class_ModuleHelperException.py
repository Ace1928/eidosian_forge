from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.text.converters import to_native
class ModuleHelperException(Exception):

    def __init__(self, msg, update_output=None, *args, **kwargs):
        self.msg = to_native(msg or 'Module failed with exception: {0}'.format(self))
        if update_output is None:
            update_output = {}
        self.update_output = update_output
        super(ModuleHelperException, self).__init__(*args)