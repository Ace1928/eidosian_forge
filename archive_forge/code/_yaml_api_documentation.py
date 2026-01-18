from typing import Union, IO, Any
from io import StringIO
import sys
from .ruamel_yaml import YAML
from .ruamel_yaml.representer import RepresenterError
from .util import force_path, FilePath, YAMLInput, YAMLOutput
Check if a Python object is YAML-serializable (strict).

    obj: The object to check.
    RETURNS (bool): Whether the object is YAML-serializable.
    