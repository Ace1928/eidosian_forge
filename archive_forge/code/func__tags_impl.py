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
def _tags_impl(self, ctx, experiment=None):
    """Returns tag metadata for a given experiment's logged metrics.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: optional string ID of the request's experiment.

        Returns:
            A nested dict 'd' with keys in ("scalars", "histograms", "images")
                and values being the return type of _format_*mapping.
        """
    scalar_mapping = self._data_provider.list_scalars(ctx, experiment_id=experiment, plugin_name=scalar_metadata.PLUGIN_NAME)
    scalar_mapping = self._filter_by_version(scalar_mapping, scalar_metadata.parse_plugin_metadata, self._scalar_version_checker)
    histogram_mapping = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=histogram_metadata.PLUGIN_NAME)
    if histogram_mapping is None:
        histogram_mapping = {}
    histogram_mapping = self._filter_by_version(histogram_mapping, histogram_metadata.parse_plugin_metadata, self._histogram_version_checker)
    image_mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=image_metadata.PLUGIN_NAME)
    if image_mapping is None:
        image_mapping = {}
    image_mapping = self._filter_by_version(image_mapping, image_metadata.parse_plugin_metadata, self._image_version_checker)
    result = {}
    result['scalars'] = _format_basic_mapping(scalar_mapping)
    result['histograms'] = _format_basic_mapping(histogram_mapping)
    result['images'] = _format_image_mapping(image_mapping)
    return result