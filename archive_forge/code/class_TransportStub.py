import os
import uuid
import xmltodict
from pytest import skip, fixture
from mock import patch
class TransportStub(object):

    def send_message(self, message):
        if xml_str_compare(message, open_shell_request):
            return open_shell_response
        elif xml_str_compare(message, close_shell_request):
            return close_shell_response
        elif xml_str_compare(message, run_cmd_with_args_request) or xml_str_compare(message, run_cmd_wo_args_request):
            return run_cmd_ps_response % '1'
        elif xml_str_compare(message, run_ps_request):
            return run_cmd_ps_response % '2'
        elif xml_str_compare(message, cleanup_cmd_request % '1') or xml_str_compare(message, cleanup_cmd_request % '2'):
            return cleanup_cmd_response
        elif xml_str_compare(message, get_cmd_ps_output_request % '1'):
            return get_cmd_output_response
        elif xml_str_compare(message, get_cmd_ps_output_request % '2'):
            return get_ps_output_response
        elif xml_str_compare(message, run_cmd_req_input):
            return run_cmd_req_input_response
        elif xml_str_compare(message, run_cmd_send_input):
            return run_cmd_send_input_response
        elif xml_str_compare(message, run_cmd_send_input_get_output):
            return run_cmd_send_input_get_output_response
        elif xml_str_compare(message, stdin_cmd_cleanup):
            return stdin_cmd_cleanup_response
        else:
            raise Exception('Message was not expected\n\n%s' % message)

    def close_session(self):
        pass