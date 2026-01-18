from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _update_required(self, robject):
    object_remote = robject['data'] if 'data' in robject else {}
    object_present = remove_aliases(self.module.params, self.metadata)
    object_present = object_present.get(self.module_level2_name, {})
    return self.is_object_difference(object_remote, object_present)