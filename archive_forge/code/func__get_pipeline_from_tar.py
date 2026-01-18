import io
import json
from itertools import islice
from typing import Any, Callable, Dict, List
import numpy as np
import pyarrow as pa
import datasets
@classmethod
def _get_pipeline_from_tar(cls, tar_path, tar_iterator):
    current_example = {}
    for filename, f in tar_iterator:
        if '.' in filename:
            example_key, field_name = filename.split('.', 1)
            if current_example and current_example['__key__'] != example_key:
                yield current_example
                current_example = {}
            current_example['__key__'] = example_key
            current_example['__url__'] = tar_path
            current_example[field_name.lower()] = f.read()
            if field_name in cls.DECODERS:
                current_example[field_name] = cls.DECODERS[field_name](current_example[field_name])
    if current_example:
        yield current_example