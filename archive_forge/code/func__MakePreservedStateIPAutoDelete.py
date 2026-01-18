from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def _MakePreservedStateIPAutoDelete(messages, auto_delete_str):
    auto_delete_map = {'never': messages.PreservedStatePreservedNetworkIp.AutoDeleteValueValuesEnum.NEVER, 'on-permanent-instance-deletion': messages.PreservedStatePreservedNetworkIp.AutoDeleteValueValuesEnum.ON_PERMANENT_INSTANCE_DELETION}
    return auto_delete_map[auto_delete_str]