from dataclasses import dataclass
import logging
import os
import sys
from typing import List, Tuple, Optional, TYPE_CHECKING
def _register_custom_datasets_serializers(serialization_context):
    try:
        import pyarrow as pa
    except ModuleNotFoundError:
        return
    _register_arrow_data_serializer(serialization_context)
    _register_arrow_json_readoptions_serializer(serialization_context)
    _register_arrow_json_parseoptions_serializer(serialization_context)