import json
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend import process_graph
from tensorboard.compat.proto import config_pb2
from tensorboard.compat.proto import graph_pb2
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.graph import graph_util
from tensorboard.plugins.graph import keras_util
from tensorboard.plugins.graph import metadata
from tensorboard.util import tb_logging
class GraphsPlugin(base_plugin.TBPlugin):
    """Graphs Plugin for TensorBoard."""
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates GraphsPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._data_provider = context.data_provider

    def get_plugin_apps(self):
        return {'/graph': self.graph_route, '/info': self.info_route, '/run_metadata': self.run_metadata_route}

    def is_active(self):
        """The graphs plugin is active iff any run has a graph or metadata."""
        return False

    def data_plugin_names(self):
        return (metadata.PLUGIN_NAME, metadata.PLUGIN_NAME_RUN_METADATA, metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH, metadata.PLUGIN_NAME_KERAS_MODEL, metadata.PLUGIN_NAME_TAGGED_RUN_METADATA)

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-graph-dashboard', disable_reload=True)

    def info_impl(self, ctx, experiment=None):
        """Returns a dict of all runs and their data availabilities."""
        result = {}

        def add_row_item(run, tag=None):
            run_item = result.setdefault(run, {'run': run, 'tags': {}, 'run_graph': False})
            tag_item = None
            if tag:
                tag_item = run_item.get('tags').setdefault(tag, {'tag': tag, 'conceptual_graph': False, 'op_graph': False, 'profile': False})
            return (run_item, tag_item)
        mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH)
        for run_name, tags in mapping.items():
            for tag, tag_data in tags.items():
                if tag_data.plugin_content != b'1':
                    logger.warning('Ignoring unrecognizable version of RunMetadata.')
                    continue
                _, tag_item = add_row_item(run_name, tag)
                tag_item['op_graph'] = True
        mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_RUN_METADATA)
        for run_name, tags in mapping.items():
            for tag, tag_data in tags.items():
                if tag_data.plugin_content != b'1':
                    logger.warning('Ignoring unrecognizable version of RunMetadata.')
                    continue
                _, tag_item = add_row_item(run_name, tag)
                tag_item['profile'] = True
                tag_item['op_graph'] = True
        mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_KERAS_MODEL)
        for run_name, tags in mapping.items():
            for tag, tag_data in tags.items():
                if tag_data.plugin_content != b'1':
                    logger.warning('Ignoring unrecognizable version of RunMetadata.')
                    continue
                _, tag_item = add_row_item(run_name, tag)
                tag_item['conceptual_graph'] = True
        mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
        for run_name, tags in mapping.items():
            if metadata.RUN_GRAPH_NAME in tags:
                run_item, _ = add_row_item(run_name, None)
                run_item['run_graph'] = True
        mapping = self._data_provider.list_blob_sequences(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME_TAGGED_RUN_METADATA)
        for run_name, tags in mapping.items():
            for tag in tags:
                _, tag_item = add_row_item(run_name, tag)
                tag_item['profile'] = True
        return result

    def _read_blob(self, ctx, experiment, plugin_names, run, tag):
        for plugin_name in plugin_names:
            blob_sequences = self._data_provider.read_blob_sequences(ctx, experiment_id=experiment, plugin_name=plugin_name, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]), downsample=1)
            blob_sequence_data = blob_sequences.get(run, {}).get(tag, ())
            try:
                blob_ref = blob_sequence_data[0].values[0]
            except IndexError:
                continue
            return self._data_provider.read_blob(ctx, blob_key=blob_ref.blob_key)
        raise errors.NotFoundError()

    def graph_impl(self, ctx, run, tag, is_conceptual, experiment=None, limit_attr_size=None, large_attrs_key=None):
        """Result of the form `(body, mime_type)`; may raise `NotFound`."""
        if is_conceptual:
            keras_model_config = json.loads(self._read_blob(ctx, experiment, [metadata.PLUGIN_NAME_KERAS_MODEL], run, tag))
            graph = keras_util.keras_model_to_graph_def(keras_model_config)
        elif tag is None:
            graph_raw = self._read_blob(ctx, experiment, [metadata.PLUGIN_NAME], run, metadata.RUN_GRAPH_NAME)
            graph = graph_pb2.GraphDef.FromString(graph_raw)
        else:
            plugins = [metadata.PLUGIN_NAME_RUN_METADATA, metadata.PLUGIN_NAME_RUN_METADATA_WITH_GRAPH]
            raw_run_metadata = self._read_blob(ctx, experiment, plugins, run, tag)
            run_metadata = config_pb2.RunMetadata.FromString(raw_run_metadata)
            graph = graph_util.merge_graph_defs([func_graph.pre_optimization_graph for func_graph in run_metadata.function_graphs])
        process_graph.prepare_graph_for_ui(graph, limit_attr_size, large_attrs_key)
        return (str(graph), 'text/x-protobuf')

    def run_metadata_impl(self, ctx, experiment, run, tag):
        """Result of the form `(body, mime_type)`; may raise `NotFound`."""
        plugins = [metadata.PLUGIN_NAME_TAGGED_RUN_METADATA, metadata.PLUGIN_NAME_RUN_METADATA]
        raw_run_metadata = self._read_blob(ctx, experiment, plugins, run, tag)
        run_metadata = config_pb2.RunMetadata.FromString(raw_run_metadata)
        return (str(run_metadata), 'text/x-protobuf')

    @wrappers.Request.application
    def info_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        info = self.info_impl(ctx, experiment)
        return http_util.Respond(request, info, 'application/json')

    @wrappers.Request.application
    def graph_route(self, request):
        """Given a single run, return the graph definition in protobuf
        format."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get('run')
        tag = request.args.get('tag')
        conceptual_arg = request.args.get('conceptual', False)
        is_conceptual = True if conceptual_arg == 'true' else False
        if run is None:
            return http_util.Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        limit_attr_size = request.args.get('limit_attr_size', None)
        if limit_attr_size is not None:
            try:
                limit_attr_size = int(limit_attr_size)
            except ValueError:
                return http_util.Respond(request, 'query parameter `limit_attr_size` must be an integer', 'text/plain', 400)
        large_attrs_key = request.args.get('large_attrs_key', None)
        try:
            result = self.graph_impl(ctx, run, tag, is_conceptual, experiment, limit_attr_size, large_attrs_key)
        except ValueError as e:
            return http_util.Respond(request, e.message, 'text/plain', code=400)
        body, mime_type = result
        return http_util.Respond(request, body, mime_type)

    @wrappers.Request.application
    def run_metadata_route(self, request):
        """Given a tag and a run, return the session.run() metadata."""
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        tag = request.args.get('tag')
        run = request.args.get('run')
        if tag is None:
            return http_util.Respond(request, 'query parameter "tag" is required', 'text/plain', 400)
        if run is None:
            return http_util.Respond(request, 'query parameter "run" is required', 'text/plain', 400)
        body, mime_type = self.run_metadata_impl(ctx, experiment, run, tag)
        return http_util.Respond(request, body, mime_type)