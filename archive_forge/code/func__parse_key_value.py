from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import exceptions
def _parse_key_value(key_value):
    split_key_value = key_value.split('=')
    if len(split_key_value) > 2:
        raise exceptions.Error('Illegal value for toleration key-value={}'.format(key_value))
    key = split_key_value[0]
    value = split_key_value[1] if len(split_key_value) == 2 else None
    operator = 'Exists' if len(split_key_value) == 1 else 'Equal'
    return (key, value, operator)