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
@classmethod
def LoadFromURI(cls, dir_uri):
    """Loads a runtime experiment config from a gs:// or file:// path.

    Args:
      dir_uri: str, A gs:// or file:// URI pointing to a folder that contains
        the file called Experiments.CONFIG_FILE

    Returns:
      Experiments, the loaded runtime experiments config.
    """
    uri = _Join(dir_uri, cls.CONFIG_FILE)
    log.debug('Loading runtimes experiment config from [%s]', uri)
    try:
        with _Read(uri) as f:
            data = yaml.load(f, file_hint=uri)
        return cls(uri, data)
    except FileReadError as e:
        raise ExperimentsError('Unable to read the runtimes experiment config: [{}], error: {}'.format(uri, e))
    except yaml.YAMLParseError as e:
        raise ExperimentsError('Unable to read the runtimes experiment config: [{}], error: {}'.format(uri, e))