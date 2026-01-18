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
def _parse_request_argument(request, proto_class):
    request_json = request.data if request.method == 'POST' else request.args.get('request')
    try:
        return json_format.Parse(request_json, proto_class())
    except (AttributeError, json_format.ParseError) as e:
        raise error.HParamsError('Expected a JSON-formatted request data of type: {}, but got {} '.format(proto_class, request_json)) from e