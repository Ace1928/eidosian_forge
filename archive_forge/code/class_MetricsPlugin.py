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
class MetricsPlugin(base_plugin.TBPlugin):
    """Metrics Plugin for TensorBoard."""
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates MetricsPlugin.

        Args:
            context: A base_plugin.TBContext instance. MetricsLoader checks that
                it contains a valid `data_provider`.
        """
        self._data_provider = context.data_provider
        sampling_hints = context.sampling_hints or {}
        self._plugin_downsampling = {'scalars': sampling_hints.get(scalar_metadata.PLUGIN_NAME, 1000), 'histograms': sampling_hints.get(histogram_metadata.PLUGIN_NAME, 51), 'images': sampling_hints.get(image_metadata.PLUGIN_NAME, 10)}
        self._scalar_version_checker = plugin_util._MetadataVersionChecker(data_kind='scalar time series', latest_known_version=0)
        self._histogram_version_checker = plugin_util._MetadataVersionChecker(data_kind='histogram time series', latest_known_version=0)
        self._image_version_checker = plugin_util._MetadataVersionChecker(data_kind='image time series', latest_known_version=0)

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(is_ng_component=True, tab_name='Time Series')

    def get_plugin_apps(self):
        return {'/tags': self._serve_tags, '/timeSeries': self._serve_time_series, '/imageData': self._serve_image_data}

    def data_plugin_names(self):
        return (scalar_metadata.PLUGIN_NAME, histogram_metadata.PLUGIN_NAME, image_metadata.PLUGIN_NAME)

    def is_active(self):
        return False

    @wrappers.Request.application
    def _serve_tags(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self._tags_impl(ctx, experiment=experiment)
        return http_util.Respond(request, index, 'application/json')

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

    def _filter_by_version(self, mapping, parse_metadata, version_checker):
        """Filter `DataProvider.list_*` output by summary metadata version."""
        result = {run: {} for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                md = parse_metadata(metadatum.plugin_content)
                if not version_checker.ok(md.version, run, tag):
                    continue
                result[run][tag] = metadatum
        return result

    @wrappers.Request.application
    def _serve_time_series(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        if request.method == 'POST':
            series_requests_string = request.form.get('requests')
        else:
            series_requests_string = request.args.get('requests')
        if not series_requests_string:
            raise errors.InvalidArgumentError("Missing 'requests' field")
        try:
            series_requests = json.loads(series_requests_string)
        except ValueError:
            raise errors.InvalidArgumentError("Unable to parse 'requests' as JSON")
        response = self._time_series_impl(ctx, experiment, series_requests)
        return http_util.Respond(request, response, 'application/json')

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

    def _create_base_response(self, series_request):
        tag = series_request.get('tag')
        run = series_request.get('run')
        plugin = series_request.get('plugin')
        sample = series_request.get('sample')
        response = {'plugin': plugin, 'tag': tag}
        if isinstance(run, str):
            response['run'] = run
        if isinstance(sample, int):
            response['sample'] = sample
        return response

    def _get_invalid_request_error(self, series_request):
        tag = series_request.get('tag')
        plugin = series_request.get('plugin')
        run = series_request.get('run')
        sample = series_request.get('sample')
        if not isinstance(tag, str):
            return 'Missing tag'
        if plugin != scalar_metadata.PLUGIN_NAME and plugin != histogram_metadata.PLUGIN_NAME and (plugin != image_metadata.PLUGIN_NAME):
            return 'Invalid plugin'
        if plugin in _SINGLE_RUN_PLUGINS and (not isinstance(run, str)):
            return 'Missing run'
        if plugin in _SAMPLED_PLUGINS and (not isinstance(sample, int)):
            return 'Missing sample'
        return None

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

    def _get_run_to_scalar_series(self, ctx, experiment, tag, runs):
        """Builds a run-to-scalar-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            runs: optional list of run names as strings.

        Returns:
            A map from string run names to `ScalarStepDatum` (see http_api.md).
        """
        mapping = self._data_provider.read_scalars(ctx, experiment_id=experiment, plugin_name=scalar_metadata.PLUGIN_NAME, downsample=self._plugin_downsampling['scalars'], run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]))
        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            values = [{'wallTime': datum.wall_time, 'step': datum.step, 'value': datum.value} for datum in tag_data[tag]]
            run_to_series[result_run] = values
        return run_to_series

    def _format_histogram_datum_bins(self, datum):
        """Formats a histogram datum's bins for client consumption.

        Args:
            datum: a DataProvider's TensorDatum.

        Returns:
            A list of `HistogramBin`s (see http_api.md).
        """
        numpy_list = datum.numpy.tolist()
        bins = [{'min': x[0], 'max': x[1], 'count': x[2]} for x in numpy_list]
        return bins

    def _get_run_to_histogram_series(self, ctx, experiment, tag, runs):
        """Builds a run-to-histogram-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            runs: optional list of run names as strings.

        Returns:
            A map from string run names to `HistogramStepDatum` (see http_api.md).
        """
        mapping = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=histogram_metadata.PLUGIN_NAME, downsample=self._plugin_downsampling['histograms'], run_tag_filter=provider.RunTagFilter(runs=runs, tags=[tag]))
        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            values = [{'wallTime': datum.wall_time, 'step': datum.step, 'bins': self._format_histogram_datum_bins(datum)} for datum in tag_data[tag]]
            run_to_series[result_run] = values
        return run_to_series

    def _get_run_to_image_series(self, ctx, experiment, tag, sample, runs):
        """Builds a run-to-image-series dict for client consumption.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            experiment: a string experiment id.
            tag: string of the requested tag.
            sample: zero-indexed integer for the requested sample.
            runs: optional list of run names as strings.

        Returns:
            A `RunToSeries` dict (see http_api.md).
        """
        mapping = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=image_metadata.PLUGIN_NAME, downsample=self._plugin_downsampling['images'], run_tag_filter=provider.RunTagFilter(runs, tags=[tag]))
        run_to_series = {}
        for result_run, tag_data in mapping.items():
            if tag not in tag_data:
                continue
            blob_sequence_datum_list = tag_data[tag]
            series = _format_image_blob_sequence_datum(blob_sequence_datum_list, sample)
            if series:
                run_to_series[result_run] = series
        return run_to_series

    @wrappers.Request.application
    def _serve_image_data(self, request):
        """Serves an individual image."""
        ctx = plugin_util.context(request.environ)
        blob_key = request.args['imageId']
        if not blob_key:
            raise errors.InvalidArgumentError("Missing 'imageId' field")
        data, content_type = self._image_data_impl(ctx, blob_key)
        return http_util.Respond(request, data, content_type)

    def _image_data_impl(self, ctx, blob_key):
        """Gets the image data for a blob key.

        Args:
            ctx: A `tensorboard.context.RequestContext` value.
            blob_key: a string identifier for a DataProvider blob.

        Returns:
            A tuple containing:
              data: a raw bytestring of the requested image's contents.
              content_type: a string HTTP content type.
        """
        data = self._data_provider.read_blob(ctx, blob_key=blob_key)
        image_type = imghdr.what(None, data)
        content_type = _IMGHDR_TO_MIMETYPE.get(image_type, _DEFAULT_IMAGE_MIMETYPE)
        return (data, content_type)