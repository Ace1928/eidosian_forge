from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _build_initial_shape(self, model):
    shape = {'type': model['type']}
    if 'documentation' in model:
        shape['documentation'] = model['documentation']
    for attr in Shape.METADATA_ATTRS:
        if attr in model:
            shape[attr] = model[attr]
    return shape