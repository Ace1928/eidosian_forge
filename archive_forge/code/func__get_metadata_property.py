from collections import defaultdict
from typing import NamedTuple, Union
from botocore.compat import OrderedDict
from botocore.exceptions import (
from botocore.utils import CachedProperty, hyphenize_service_id, instance_cache
def _get_metadata_property(self, name):
    try:
        return self.metadata[name]
    except KeyError:
        raise UndefinedModelAttributeError(f'"{name}" not defined in the metadata of the model: {self}')