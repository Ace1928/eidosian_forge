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
class HParamsPlugin(base_plugin.TBPlugin):
    """HParams Plugin for TensorBoard.

    It supports both GETs and POSTs. See 'http_api.md' for more details.
    """
    plugin_name = metadata.PLUGIN_NAME

    def __init__(self, context):
        """Instantiates HParams plugin via TensorBoard core.

        Args:
          context: A base_plugin.TBContext instance.
        """
        self._context = backend_context.Context(context)

    def get_plugin_apps(self):
        """See base class."""
        return {'/download_data': self.download_data_route, '/experiment': self.get_experiment_route, '/session_groups': self.list_session_groups_route, '/metric_evals': self.list_metric_evals_route}

    def is_active(self):
        return False

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(element_name='tf-hparams-dashboard')

    @wrappers.Request.application
    def download_data_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            response_format = request.args.get('format')
            columns_visibility = json.loads(request.args.get('columnsVisibility'))
            request_proto = _parse_request_argument(request, api_pb2.ListSessionGroupsRequest)
            session_groups = list_session_groups.Handler(ctx, self._context, experiment_id, request_proto).run()
            experiment = get_experiment.Handler(ctx, self._context, experiment_id).run()
            body, mime_type = download_data.Handler(self._context, experiment, session_groups, response_format, columns_visibility).run()
            return http_util.Respond(request, body, mime_type)
        except error.HParamsError as e:
            logger.error('HParams error: %s' % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    @wrappers.Request.application
    def get_experiment_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            _ = _parse_request_argument(request, api_pb2.GetExperimentRequest)
            return http_util.Respond(request, json_format.MessageToJson(get_experiment.Handler(ctx, self._context, experiment_id).run(), including_default_value_fields=True), 'application/json')
        except error.HParamsError as e:
            logger.error('HParams error: %s' % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    @wrappers.Request.application
    def list_session_groups_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(request, api_pb2.ListSessionGroupsRequest)
            return http_util.Respond(request, json_format.MessageToJson(list_session_groups.Handler(ctx, self._context, experiment_id, request_proto).run(), including_default_value_fields=True), 'application/json')
        except error.HParamsError as e:
            logger.error('HParams error: %s' % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    @wrappers.Request.application
    def list_metric_evals_route(self, request):
        ctx = plugin_util.context(request.environ)
        experiment_id = plugin_util.experiment_id(request.environ)
        try:
            request_proto = _parse_request_argument(request, api_pb2.ListMetricEvalsRequest)
            scalars_plugin = self._get_scalars_plugin()
            if not scalars_plugin:
                raise werkzeug.exceptions.NotFound('Scalars plugin not loaded')
            return http_util.Respond(request, list_metric_evals.Handler(ctx, request_proto, scalars_plugin, experiment_id).run(), 'application/json')
        except error.HParamsError as e:
            logger.error('HParams error: %s' % e)
            raise werkzeug.exceptions.BadRequest(description=str(e))

    def _get_scalars_plugin(self):
        """Tries to get the scalars plugin.

        Returns:
        The scalars plugin or None if it is not yet registered.
        """
        return self._context.tb_context.plugin_name_to_instance.get(scalars_metadata.PLUGIN_NAME)