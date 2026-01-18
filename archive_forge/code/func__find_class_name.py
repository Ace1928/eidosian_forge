import collections
from enum import Enum
import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Union
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
def _find_class_name(obj, allowed_module_path_prefix: str, whitelist: Set[str]):
    """Find the class name of the object. If the object is not
    under `allowed_module_path_prefix` or if its class is not in the whitelist,
    return "Custom".

    Args:
        obj: The object under inspection.
        allowed_module_path_prefix: If the `obj`'s class is not under
            the `allowed_module_path_prefix`, its class name will be anonymized.
        whitelist: If the `obj`'s class is not in the `whitelist`,
            it will be anonymized.
    Returns:
        The class name to be tagged with telemetry.
    """
    module_path = obj.__module__
    cls_name = obj.__class__.__name__
    if module_path.startswith(allowed_module_path_prefix) and cls_name in whitelist:
        return cls_name
    else:
        return 'Custom'