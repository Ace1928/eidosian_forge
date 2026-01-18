from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.recommender import service as recommender_service
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.calliope import base
from googlecloudsdk.core import yaml
def ReadConfig(config_file, message_type):
    """Parses json config file.

  Args:
    config_file: file path of the config file.
    message_type: The protorpc Message type.

  Returns:
    A message of type "message_type".
  """
    config = None
    data = yaml.load_path(config_file)
    if data:
        config = messages_util.DictToMessageWithErrorCheck(data, message_type)
    return config