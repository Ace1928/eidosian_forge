from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
class _ImportSyntheticCompositeTypeFile(_BaseImport):
    """Performs common operations on an imported composite type."""

    def __init__(self, full_path, properties=None):
        name = full_path.split(':')[1]
        super(_ImportSyntheticCompositeTypeFile, self).__init__(full_path, name)
        self.properties = properties

    def GetBaseName(self):
        if self.base_name is None:
            self.base_name = self.name
        return self.base_name

    def Exists(self):
        return True

    def GetContent(self):
        """Returns the content of the synthetic file as a string."""
        if self.content is None:
            resources = {'resources': [{'type': self.full_path, 'name': self.name}]}
            if self.properties:
                resources['resources'][0]['properties'] = self.properties
            self.content = yaml.dump(resources)
        return self.content

    def BuildChildPath(self, unused_child_path):
        raise NotImplementedError