from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from apitools.base.py import encoding
from googlecloudsdk.api_lib.policy_intelligence import orgpolicy_simulator
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def GetCustomConstraintMessageFromFile(filepath, release_track):
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
    op_simulator_api = orgpolicy_simulator.OrgPolicySimulatorApi(release_track)
    message = getattr(op_simulator_api.messages, _GetCustomConstraintMessage())
    try:
        return encoding.JsonToMessage(message, json_str)
    except Exception as e:
        raise exceptions.BadFileException('Unable to parse file [{}]: {}.'.format(filepath, e))