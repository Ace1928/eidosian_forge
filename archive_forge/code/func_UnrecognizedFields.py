from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def UnrecognizedFields(message):
    unrecognized_fields = message.all_unrecognized_fields()
    if unrecognized_fields:
        raise cloudbuild_exceptions.InvalidYamlError('Unrecognized fields in yaml: {f}'.format(f=', '.join(unrecognized_fields)))