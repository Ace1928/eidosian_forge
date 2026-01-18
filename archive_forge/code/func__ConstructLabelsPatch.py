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
def _ConstructLabelsPatch(clear_labels, remove_labels, update_labels, release_track=base.ReleaseTrack.GA):
    """Constructs an environment patch for updating labels.

  Args:
    clear_labels: bool, whether to clear the labels dictionary.
    remove_labels: iterable(string), Iterable of label names to remove.
    update_labels: {string: string}, dict of label names and values to set.
    release_track: base.ReleaseTrack, the release track of command. Will dictate
      which Composer client library will be used.

  Returns:
    (str, Environment), the field mask and environment to use for update.
  """
    messages = api_util.GetMessagesModule(release_track=release_track)
    env_cls = messages.Environment
    entry_cls = env_cls.LabelsValue.AdditionalProperty

    def _BuildEnv(entries):
        return env_cls(labels=env_cls.LabelsValue(additionalProperties=entries))
    return command_util.BuildPartialUpdate(clear_labels, remove_labels, update_labels, 'labels', entry_cls, _BuildEnv)