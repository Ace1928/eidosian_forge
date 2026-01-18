from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import logging
import os
import subprocess
import sys
import threading
from . import comm
import ruamel.yaml as yaml
from six.moves import input
def _ProcessMessage(self, plugin_stdin, message, result, params, runtime_data):
    """Process a message received from the plugin.

    Args:
      plugin_stdin: (file) The standard input stream of the plugin process.
      message: ({str: object, ...}) The message (this maps directly to the
        message's json object).
      result: (PluginResult) A result object in which to store data collected
        from some types of message.
      params: (Params) Parameters passed in through the
        fingerprinter.
      runtime_data: (object or None) Arbitrary runtime data obtained from the
        "detect" plugin.  This will be None if we are processing a message for
        the detect plugin itself or if no runtime data was provided.
    """

    def SendResponse(response):
        json.dump(response, plugin_stdin)
        plugin_stdin.write('\n')
        plugin_stdin.flush()
    msg_type = message.get('type')
    if msg_type is None:
        logging.error('Missing type in message: %0.80s' % str(message))
    elif msg_type in _LOG_FUNCS:
        _LOG_FUNCS[msg_type](message.get('message'))
    elif msg_type == 'runtime_parameters':
        try:
            result.runtime_data = message['runtime_data']
        except KeyError:
            logging.error(_MISSING_FIELD_ERROR.format('runtime_data', msg_type))
        result.generated_appinfo = message.get('appinfo')
    elif msg_type == 'gen_file':
        try:
            filename = message['filename']
            contents = message['contents']
            result.files.append(GeneratedFile(filename, contents))
        except KeyError as ex:
            logging.error(_MISSING_FIELD_ERROR.format(ex, msg_type))
    elif msg_type == 'get_config':
        response = {'type': 'get_config_response', 'params': params.ToDict(), 'runtime_data': runtime_data}
        SendResponse(response)
    elif msg_type == 'query_user':
        try:
            prompt = message['prompt']
        except KeyError as ex:
            logging.error(_MISSING_FIELD_ERROR.format('prompt', msg_type))
            return
        default = message.get('default')
        if self.env.CanPrompt():
            if default:
                message = '{0} [{1}]: '.format(prompt, default)
            else:
                message = prompt + ':'
            result = self.env.PromptResponse(message)
        elif default is not None:
            result = default
        else:
            result = ''
            logging.error(_NO_DEFAULT_ERROR.format(prompt))
        SendResponse({'type': 'query_user_response', 'result': result})
    elif msg_type == 'set_docker_context':
        try:
            result.docker_context = message['path']
        except KeyError:
            logging.error(_MISSING_FIELD_ERROR.format('path', msg_type))
            return
    else:
        logging.error('Unknown message type %s' % msg_type)