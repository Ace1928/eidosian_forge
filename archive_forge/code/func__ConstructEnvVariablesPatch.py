from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.composer import environments_util as environments_api_util
from googlecloudsdk.api_lib.composer import operations_util as operations_api_util
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import util as command_util
from googlecloudsdk.core import log
import six
def _ConstructEnvVariablesPatch(env_ref, clear_env_variables, remove_env_variables, update_env_variables, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for updating environment variables.

  Note that environment variable updates do not support partial update masks
  unlike other map updates due to comments in (b/78298321). For this reason, we
  need to retrieve the Environment, apply an update on EnvVariable dictionary,
  and patch the entire dictionary. The potential race condition here
  (environment variables being updated between when we retrieve them and when we
  send patch request)is not a concern since environment variable updates take
  5 mins to complete, and environments cannot be updated while already in the
  updating state.

  Args:
    env_ref: resource argument, Environment resource argument for environment
      being updated.
    clear_env_variables: bool, whether to clear the environment variables
      dictionary.
    remove_env_variables: iterable(string), Iterable of environment variable
      names to remove.
    update_env_variables: {string: string}, dict of environment variable names
      and values to set.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    env_obj = environments_api_util.Get(env_ref, release_track=release_track)
    initial_env_var_value = env_obj.config.softwareConfig.envVariables
    initial_env_var_list = initial_env_var_value.additionalProperties if initial_env_var_value else []
    messages = api_util.GetMessagesModule(release_track=release_track)
    env_cls = messages.Environment
    env_variables_cls = messages.SoftwareConfig.EnvVariablesValue
    entry_cls = env_variables_cls.AdditionalProperty

    def _BuildEnv(entries):
        software_config = messages.SoftwareConfig(envVariables=env_variables_cls(additionalProperties=entries))
        config = messages.EnvironmentConfig(softwareConfig=software_config)
        return env_cls(config=config)
    return ('config.software_config.env_variables', command_util.BuildFullMapUpdate(clear_env_variables, remove_env_variables, update_env_variables, initial_env_var_list, entry_cls, _BuildEnv))