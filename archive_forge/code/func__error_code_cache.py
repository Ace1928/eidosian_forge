from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
@CachedProperty
def _error_code_cache(self):
    error_code_cache = {}
    for error_shape in self.error_shapes:
        code = error_shape.error_code
        error_code_cache[code] = error_shape
    return error_code_cache