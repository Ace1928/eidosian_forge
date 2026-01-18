from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import json
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
import six
@property
def _base_url(self):
    return self._sts_template.format(service=self._service, mtls=self._mtls, universe=self._universe_domain)