from __future__ import absolute_import, division, print_function
import sys
import json
import re
import traceback as trace
def _extract_rest_call_site(traceback):
    while traceback:
        try:
            frame = traceback.tb_frame
            func_name = frame.f_code.co_name
            mod_path = frame.f_globals['__name__']
            path_segments = mod_path.split('.')
            if path_segments[0] == 'fusion' and path_segments[1] == 'api' and ('_api' in path_segments[2]):
                call_site = func_name.replace('_', ' ').capitalize()
                return call_site
        except Exception:
            pass
        traceback = traceback.tb_next
    return None