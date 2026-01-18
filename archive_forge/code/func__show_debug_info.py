from _pydevd_bundle.pydevd_constants import DebugInfoHolder, \
from _pydevd_bundle.pydevd_utils import quote_smart as quote, to_string
from _pydevd_bundle.pydevd_comm_constants import ID_TO_MEANING, CMD_EXIT
from _pydevd_bundle.pydevd_constants import HTTP_PROTOCOL, HTTP_JSON_PROTOCOL, \
import json
from _pydev_bundle import pydev_log
@classmethod
def _show_debug_info(cls, cmd_id, seq, text):
    with cls._show_debug_info_lock:
        if cls._showing_debug_info:
            return
        cls._showing_debug_info += 1
        try:
            out_message = 'sending cmd (%s) --> ' % (get_protocol(),)
            out_message += '%20s' % ID_TO_MEANING.get(str(cmd_id), 'UNKNOWN')
            out_message += ' '
            out_message += text.replace('\n', ' ')
            try:
                pydev_log.critical('%s\n', out_message)
            except:
                pass
        finally:
            cls._showing_debug_info -= 1