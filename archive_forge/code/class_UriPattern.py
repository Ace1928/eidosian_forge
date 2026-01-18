from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import base64
import json
import re
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
import six
class UriPattern(object):
    """A uri-based pattern that maybe be matched against resource objects."""

    def __init__(self, path_as_string):
        if not path_as_string.startswith('http'):
            raise BadPatternException('uri', path_as_string)
        self._path_as_string = resources.REGISTRY.ParseURL(path_as_string).RelativeName()

    def Matches(self, resource):
        """Tests if its argument matches the pattern."""
        return self._path_as_string == resource.RelativeName()

    def __str__(self):
        return 'Uri Pattern: ' + self._path_as_string