import collections
import contextlib
import copy
import inspect
import json
import sys
import textwrap
from typing import (
from itertools import zip_longest
from importlib.metadata import version as importlib_version
from typing import Final
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
import numpy as np
import pandas as pd
from packaging.version import Version
from altair import vegalite
def _get_referencing_registry(rootschema: Dict[str, Any], json_schema_draft_url: Optional[str]=None):
    import referencing
    import referencing.jsonschema
    if json_schema_draft_url is None:
        json_schema_draft_url = _get_json_schema_draft_url(rootschema)
    specification = referencing.jsonschema.specification_with(json_schema_draft_url)
    resource = specification.create_resource(rootschema)
    return referencing.Registry().with_resource(uri=_VEGA_LITE_ROOT_URI, resource=resource)