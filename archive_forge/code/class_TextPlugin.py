import textwrap
import numpy as np
from werkzeug import wrappers
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.data import provider
from tensorboard.plugins import base_plugin
from tensorboard.plugins.text import metadata
class TextPlugin(base_plugin.TBPlugin):
    """Text Plugin for TensorBoard."""
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates TextPlugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._downsample_to = (context.sampling_hints or {}).get(self.plugin_name, _DEFAULT_DOWNSAMPLING)
        self._data_provider = context.data_provider
        self._version_checker = plugin_util._MetadataVersionChecker(data_kind='text', latest_known_version=0)

    def is_active(self):
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-text-dashboard')

    def index_impl(self, ctx, experiment):
        mapping = self._data_provider.list_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME)
        result = {run: [] for run in mapping}
        for run, tag_to_content in mapping.items():
            for tag, metadatum in tag_to_content.items():
                md = metadata.parse_plugin_metadata(metadatum.plugin_content)
                if not self._version_checker.ok(md.version, run, tag):
                    continue
                result[run].append(tag)
        return result

    @wrappers.Request.application
    def tags_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        index = self.index_impl(ctx, experiment)
        return http_util.Respond(request, index, 'application/json')

    def text_impl(self, ctx, run, tag, experiment, enable_markdown):
        all_text = self._data_provider.read_tensors(ctx, experiment_id=experiment, plugin_name=metadata.PLUGIN_NAME, downsample=self._downsample_to, run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]))
        text = all_text.get(run, {}).get(tag, None)
        if text is None:
            return []
        return [process_event(d.wall_time, d.step, d.numpy, enable_markdown) for d in text]

    @wrappers.Request.application
    def text_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        run = request.args.get('run')
        tag = request.args.get('tag')
        markdown_arg = request.args.get('markdown')
        enable_markdown = markdown_arg != 'false'
        response = self.text_impl(ctx, run, tag, experiment, enable_markdown)
        return http_util.Respond(request, response, 'application/json')

    def get_plugin_apps(self):
        return {TAGS_ROUTE: self.tags_route, TEXT_ROUTE: self.text_route}