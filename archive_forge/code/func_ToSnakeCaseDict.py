from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from dateutil import tz
from googlecloudsdk.core.util import times
def ToSnakeCaseDict(dictionary):
    """Recursively convert all keys in nested dictionaries to snakeCase."""
    new_dict = {}
    for key, val in dictionary.items():
        snaked_key = SnakeCaseToCamelCase(key)
        if isinstance(val, dict):
            new_dict[snaked_key] = ToSnakeCaseDict(val)
        else:
            new_dict[snaked_key] = val
    return new_dict