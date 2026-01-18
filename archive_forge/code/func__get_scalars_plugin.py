import json
import werkzeug
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.plugins.hparams import api_pb2
from tensorboard.plugins.hparams import backend_context
from tensorboard.plugins.hparams import download_data
from tensorboard.plugins.hparams import error
from tensorboard.plugins.hparams import get_experiment
from tensorboard.plugins.hparams import list_metric_evals
from tensorboard.plugins.hparams import list_session_groups
from tensorboard.plugins.hparams import metadata
from google.protobuf import json_format
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.scalar import metadata as scalars_metadata
from tensorboard.util import tb_logging
def _get_scalars_plugin(self):
    """Tries to get the scalars plugin.

        Returns:
        The scalars plugin or None if it is not yet registered.
        """
    return self._context.tb_context.plugin_name_to_instance.get(scalars_metadata.PLUGIN_NAME)