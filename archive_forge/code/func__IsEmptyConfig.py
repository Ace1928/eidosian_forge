from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
def _IsEmptyConfig(config):
    if config is None:
        return True
    config_dict = encoding.MessageToDict(config)
    return not any(config_dict.values())