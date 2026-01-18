from source directory to docker image. They are stored as templated .yaml files
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import contextlib
import enum
import os
import re
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.api_lib.cloudbuild import config as cloudbuild_config
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
import six
import six.moves.urllib.error
import six.moves.urllib.parse
import six.moves.urllib.request
def _GetAllowlist(self):
    """Return the allowlist of runtimes for this strategy.

    The allowlist is kept as a constant within this module.

    Returns:
      list of str, the names of runtimes that are allowed for this strategy.

    Raises:
      ValueError: if this strategy is not allowlist-based.
    """
    if self is self.ALLOWLIST_GA:
        return _ALLOWLISTED_RUNTIMES_GA
    elif self is self.ALLOWLIST_BETA:
        return _ALLOWLISTED_RUNTIMES_BETA
    raise ValueError('RuntimeBuilderStrategy {} is not an allowed strategy.'.format(self))