import collections
import json
import os
from ._py_components_generation import (
from .base_component import ComponentRegistry
Load React component metadata into a format Dash can parse, then create
    Python class files.

    Usage: generate_classes()

    Keyword arguments:
    namespace -- name of the generated Python package (also output dir)

    metadata_path -- a path to a JSON file created by
    [`react-docgen`](https://github.com/reactjs/react-docgen).

    Returns:
    