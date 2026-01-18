from decimal import (
from boto3.compat import collections_abc
def _is_listlike(self, value):
    if isinstance(value, (list, tuple)):
        return True
    return False