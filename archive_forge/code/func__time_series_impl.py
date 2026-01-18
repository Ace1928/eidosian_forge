import collections
import imghdr
import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.histogram import metadata as histogram_metadata
from tensorboard.plugins.image import metadata as image_metadata
from tensorboard.plugins.metrics import metadata
from tensorboard.plugins.scalar import metadata as scalar_metadata
def _time_series_impl(self, ctx, experiment, series_requests):
    """Constructs a list of responses from a list of series requests.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: string ID of the request's experiment.
            series_requests: a list of `TimeSeriesRequest` dicts (see http_api.md).

        Returns:
            A list of `TimeSeriesResponse` dicts (see http_api.md).
        """
    responses = [self._get_time_series(ctx, experiment, request) for request in series_requests]
    return responses