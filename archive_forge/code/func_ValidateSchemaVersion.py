from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import pkg_resources
import jsonschema
def ValidateSchemaVersion(self, schema, path):
    """Validates the parsed_yaml JSON schema version."""
    try:
        version = schema.get('$schema')
    except AttributeError:
        version = None
    if not version or not version.startswith('http://json-schema.org/') or (not version.endswith('/schema#')):
        raise InvalidSchemaVersionError('Schema [{}] version [{}] is invalid. Expected "$schema: http://json-schema.org/*/schema#".'.format(path, version))