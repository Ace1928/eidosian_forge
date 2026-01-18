import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.data import provider
from tensorboard.backend import http_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.pr_curve import metadata
def _process_datum(self, datum):
    """Converts a TensorDatum into a dict that encapsulates information on
        it.

        Args:
          datum: The TensorDatum to convert.

        Returns:
          A JSON-able dictionary of PR curve data for 1 step.
        """
    return self._make_pr_entry(datum.step, datum.wall_time, datum.numpy)