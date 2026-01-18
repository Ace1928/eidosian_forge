import threading
from werkzeug import wrappers
from tensorboard import errors
from tensorboard import plugin_util
from tensorboard.plugins import base_plugin
from tensorboard.plugins.debugger_v2 import debug_data_provider
from tensorboard.backend import http_util
def _missing_run_error_response(request):
    return _error_response(request, 'run parameter is not provided')