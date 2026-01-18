import csv
import datetime
import json
import logging
import apache_beam as beam
from six.moves import cStringIO
import yaml
from google.cloud.ml.util import _decoders
from google.cloud.ml.util import _file
class YamlCoder(beam.coders.Coder):
    """A coder to encode and decode YAML formatted data."""

    def __init__(self):
        """Trying to use the efficient libyaml library to encode and decode yaml.

    If libyaml is not available than we fallback to use the native yaml library,
    use with caution; it is far less efficient, uses excessive memory, and leaks
    memory.
    """
        if yaml.__with_libyaml__:
            self._safe_dumper = yaml.CSafeDumper
            self._safe_loader = yaml.CSafeLoader
        else:
            logging.warning("Can't find libyaml so it is not used for YamlCoder, the implementation used is far slower and has a memory leak.")
            self._safe_dumper = yaml.SafeDumper
            self._safe_loader = yaml.SafeLoader

    def encode(self, obj):
        """Encodes a python object into a YAML string.

    Args:
      obj: python object.

    Returns:
      YAML string.
    """
        return yaml.dump(obj, default_flow_style=False, encoding='utf-8', Dumper=self._safe_dumper)

    def decode(self, yaml_string):
        """Decodes a YAML string to a python object.

    Args:
      yaml_string: A YAML string.

    Returns:
      A python object.
    """
        return yaml.load(yaml_string, Loader=self._safe_loader)