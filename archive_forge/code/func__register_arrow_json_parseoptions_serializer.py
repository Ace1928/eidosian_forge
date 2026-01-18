from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _register_arrow_json_parseoptions_serializer(serialization_context):
    if os.environ.get(RAY_DISABLE_CUSTOM_ARROW_JSON_OPTIONS_SERIALIZATION, '0') == '1':
        return
    import pyarrow.json as pajson
    serialization_context._register_cloudpickle_serializer(pajson.ParseOptions, custom_serializer=lambda opts: (opts.explicit_schema, opts.newlines_in_values, opts.unexpected_field_behavior), custom_deserializer=lambda args: pajson.ParseOptions(*args))