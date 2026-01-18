import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _SnakeCaseToCamelCase(path_name):
    """Converts a path name from snake_case to camelCase."""
    result = []
    after_underscore = False
    for c in path_name:
        if c.isupper():
            raise ValueError('Fail to print FieldMask to Json string: Path name {0} must not contain uppercase letters.'.format(path_name))
        if after_underscore:
            if c.islower():
                result.append(c.upper())
                after_underscore = False
            else:
                raise ValueError('Fail to print FieldMask to Json string: The character after a "_" must be a lowercase letter in path name {0}.'.format(path_name))
        elif c == '_':
            after_underscore = True
        else:
            result += c
    if after_underscore:
        raise ValueError('Fail to print FieldMask to Json string: Trailing "_" in path name {0}.'.format(path_name))
    return ''.join(result)