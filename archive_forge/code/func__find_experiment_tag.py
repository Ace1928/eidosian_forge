import collections
import os
from tensorboard.data import provider
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import json_format_compat
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _find_experiment_tag(self, hparams_run_to_tag_to_content):
    """Finds the experiment associcated with the metadata.EXPERIMENT_TAG
        tag.

        Returns:
          The experiment or None if no such experiment is found.
        """
    for tags in hparams_run_to_tag_to_content.values():
        maybe_content = tags.get(metadata.EXPERIMENT_TAG)
        if maybe_content is not None:
            return metadata.parse_experiment_plugin_data(maybe_content)
    return None