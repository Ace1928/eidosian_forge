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
def _get_time_series(self, ctx, experiment, series_request):
    """Returns time series data for a given tag, plugin.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: string ID of the request's experiment.
            series_request: a `TimeSeriesRequest` (see http_api.md).

        Returns:
            A `TimeSeriesResponse` dict (see http_api.md).
        """
    tag = series_request.get('tag')
    run = series_request.get('run')
    plugin = series_request.get('plugin')
    sample = series_request.get('sample')
    response = self._create_base_response(series_request)
    request_error = self._get_invalid_request_error(series_request)
    if request_error:
        response['error'] = request_error
        return response
    runs = [run] if run else None
    run_to_series = None
    if plugin == scalar_metadata.PLUGIN_NAME:
        run_to_series = self._get_run_to_scalar_series(ctx, experiment, tag, runs)
    if plugin == histogram_metadata.PLUGIN_NAME:
        run_to_series = self._get_run_to_histogram_series(ctx, experiment, tag, runs)
    if plugin == image_metadata.PLUGIN_NAME:
        run_to_series = self._get_run_to_image_series(ctx, experiment, tag, sample, runs)
    response['runToSeries'] = run_to_series
    return response