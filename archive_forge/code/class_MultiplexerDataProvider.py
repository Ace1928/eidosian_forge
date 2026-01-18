import base64
import json
import random
from tensorboard import errors
from tensorboard.compat.proto import summary_pb2
from tensorboard.data import provider
from tensorboard.util import tb_logging
from tensorboard.util import tensor_util
class MultiplexerDataProvider(provider.DataProvider):

    def __init__(self, multiplexer, logdir):
        """Trivial initializer.

        Args:
          multiplexer: A `plugin_event_multiplexer.EventMultiplexer` (note:
            not a boring old `event_multiplexer.EventMultiplexer`).
          logdir: The log directory from which data is being read. Only used
            cosmetically. Should be a `str`.
        """
        self._multiplexer = multiplexer
        self._logdir = logdir

    def __str__(self):
        return 'MultiplexerDataProvider(logdir=%r)' % self._logdir

    def _validate_context(self, ctx):
        if type(ctx).__name__ != 'RequestContext':
            raise TypeError('ctx must be a RequestContext; got: %r' % (ctx,))

    def _validate_experiment_id(self, experiment_id):
        if not isinstance(experiment_id, str):
            raise TypeError('experiment_id must be %r, but got %r: %r' % (str, type(experiment_id), experiment_id))

    def _validate_downsample(self, downsample):
        if downsample is None:
            raise TypeError('`downsample` required but not given')
        if isinstance(downsample, int):
            return
        raise TypeError('`downsample` must be an int, but got %r: %r' % (type(downsample), downsample))

    def _test_run_tag(self, run_tag_filter, run, tag):
        runs = run_tag_filter.runs
        if runs is not None and run not in runs:
            return False
        tags = run_tag_filter.tags
        if tags is not None and tag not in tags:
            return False
        return True

    def _get_first_event_timestamp(self, run_name):
        try:
            return self._multiplexer.FirstEventTimestamp(run_name)
        except ValueError as e:
            return None

    def experiment_metadata(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        return provider.ExperimentMetadata(data_location=self._logdir)

    def list_plugins(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        return self._multiplexer.ActivePlugins()

    def list_runs(self, ctx=None, *, experiment_id):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        return [provider.Run(run_id=run, run_name=run, start_time=self._get_first_event_timestamp(run)) for run in self._multiplexer.Runs()]

    def list_scalars(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_SCALAR)
        return self._list(provider.ScalarTimeSeries, index)

    def read_scalars(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_SCALAR)
        return self._read(_convert_scalar_event, index, downsample)

    def list_tensors(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_TENSOR)
        return self._list(provider.TensorTimeSeries, index)

    def read_tensors(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_TENSOR)
        return self._read(_convert_tensor_event, index, downsample)

    def _index(self, plugin_name, run_tag_filter, data_class_filter):
        """List time series and metadata matching the given filters.

        This is like `_list`, but doesn't traverse `Tensors(...)` to
        compute metadata that's not always needed.

        Args:
          plugin_name: A string plugin name filter (required).
          run_tag_filter: An `provider.RunTagFilter`, or `None`.
          data_class_filter: A `summary_pb2.DataClass` filter (required).

        Returns:
          A nested dict `d` such that `d[run][tag]` is a
          `SummaryMetadata` proto.
        """
        if run_tag_filter is None:
            run_tag_filter = provider.RunTagFilter(runs=None, tags=None)
        runs = run_tag_filter.runs
        tags = run_tag_filter.tags
        if runs and len(runs) == 1 and tags and (len(tags) == 1):
            run, = runs
            tag, = tags
            try:
                metadata = self._multiplexer.SummaryMetadata(run, tag)
            except KeyError:
                return {}
            all_metadata = {run: {tag: metadata}}
        else:
            all_metadata = self._multiplexer.AllSummaryMetadata()
        result = {}
        for run, tag_to_metadata in all_metadata.items():
            if runs is not None and run not in runs:
                continue
            result_for_run = {}
            for tag, metadata in tag_to_metadata.items():
                if tags is not None and tag not in tags:
                    continue
                if metadata.data_class != data_class_filter:
                    continue
                if metadata.plugin_data.plugin_name != plugin_name:
                    continue
                result[run] = result_for_run
                result_for_run[tag] = metadata
        return result

    def _list(self, construct_time_series, index):
        """Helper to list scalar or tensor time series.

        Args:
          construct_time_series: `ScalarTimeSeries` or `TensorTimeSeries`.
          index: The result of `self._index(...)`.

        Returns:
          A list of objects of type given by `construct_time_series`,
          suitable to be returned from `list_scalars` or `list_tensors`.
        """
        result = {}
        for run, tag_to_metadata in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, summary_metadata in tag_to_metadata.items():
                max_step = None
                max_wall_time = None
                for event in self._multiplexer.Tensors(run, tag):
                    if max_step is None or max_step < event.step:
                        max_step = event.step
                    if max_wall_time is None or max_wall_time < event.wall_time:
                        max_wall_time = event.wall_time
                summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
                result_for_run[tag] = construct_time_series(max_step=max_step, max_wall_time=max_wall_time, plugin_content=summary_metadata.plugin_data.content, description=summary_metadata.summary_description, display_name=summary_metadata.display_name)
        return result

    def _read(self, convert_event, index, downsample):
        """Helper to read scalar or tensor data from the multiplexer.

        Args:
          convert_event: Takes `plugin_event_accumulator.TensorEvent` to
            either `provider.ScalarDatum` or `provider.TensorDatum`.
          index: The result of `self._index(...)`.
          downsample: Non-negative `int`; how many samples to return per
            time series.

        Returns:
          A dict of dicts of values returned by `convert_event` calls,
          suitable to be returned from `read_scalars` or `read_tensors`.
        """
        result = {}
        for run, tags_for_run in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, metadata in tags_for_run.items():
                events = self._multiplexer.Tensors(run, tag)
                data = [convert_event(e) for e in events]
                result_for_run[tag] = _downsample(data, downsample)
        return result

    def list_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_BLOB_SEQUENCE)
        result = {}
        for run, tag_to_metadata in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag, metadata in tag_to_metadata.items():
                max_step = None
                max_wall_time = None
                max_length = None
                for event in self._multiplexer.Tensors(run, tag):
                    if max_step is None or max_step < event.step:
                        max_step = event.step
                    if max_wall_time is None or max_wall_time < event.wall_time:
                        max_wall_time = event.wall_time
                    length = _tensor_size(event.tensor_proto)
                    if max_length is None or length > max_length:
                        max_length = length
                result_for_run[tag] = provider.BlobSequenceTimeSeries(max_step=max_step, max_wall_time=max_wall_time, max_length=max_length, plugin_content=metadata.plugin_data.content, description=metadata.summary_description, display_name=metadata.display_name)
        return result

    def read_blob_sequences(self, ctx=None, *, experiment_id, plugin_name, downsample=None, run_tag_filter=None):
        self._validate_context(ctx)
        self._validate_experiment_id(experiment_id)
        self._validate_downsample(downsample)
        index = self._index(plugin_name, run_tag_filter, summary_pb2.DATA_CLASS_BLOB_SEQUENCE)
        result = {}
        for run, tags in index.items():
            result_for_run = {}
            result[run] = result_for_run
            for tag in tags:
                events = self._multiplexer.Tensors(run, tag)
                data_by_step = {}
                for event in events:
                    if event.step in data_by_step:
                        continue
                    data_by_step[event.step] = _convert_blob_sequence_event(experiment_id, plugin_name, run, tag, event)
                data = [datum for step, datum in sorted(data_by_step.items())]
                result_for_run[tag] = _downsample(data, downsample)
        return result

    def read_blob(self, ctx=None, *, blob_key):
        self._validate_context(ctx)
        unused_experiment_id, plugin_name, run, tag, step, index = _decode_blob_key(blob_key)
        summary_metadata = self._multiplexer.SummaryMetadata(run, tag)
        if summary_metadata.data_class != summary_pb2.DATA_CLASS_BLOB_SEQUENCE:
            raise errors.NotFoundError(blob_key)
        tensor_events = self._multiplexer.Tensors(run, tag)
        matching_step = next((e for e in tensor_events if e.step == step), None)
        if not matching_step:
            raise errors.NotFoundError('%s: no such step %r' % (blob_key, step))
        tensor = tensor_util.make_ndarray(matching_step.tensor_proto)
        return tensor[index]