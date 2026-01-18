from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.orgpolicy import service as org_policy_service
from googlecloudsdk.command_lib.org_policies import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetMessageFromFile(filepath, release_track):
    """Returns a message populated from the JSON or YAML file on the specified filepath.

  Args:
    filepath: str, A local path to an object specification in JSON or YAML
      format.
    release_track: calliope.base.ReleaseTrack, Release track of the command.
  """
    file_contents = files.ReadFileContents(filepath)
    try:
        yaml_obj = yaml.load(file_contents)
        json_str = json.dumps(yaml_obj)
    except yaml.YAMLParseError:
        json_str = file_contents
    org_policy_messages = org_policy_service.OrgPolicyMessages(release_track)
    message = getattr(org_policy_messages, _GetPolicyMessageName(release_track))
    try:
        return encoding.JsonToMessage(message, json_str)
    except Exception as e:
        raise exceptions.InvalidInputError('Unable to parse file [{}]: {}.'.format(filepath, e))