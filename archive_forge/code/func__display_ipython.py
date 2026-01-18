import datetime
import errno
import html
import json
import os
import random
import shlex
import textwrap
import time
from tensorboard import manager
def _display_ipython(port, height, display_handle):
    import IPython.display
    frame_id = 'tensorboard-frame-{:08x}'.format(random.getrandbits(64))
    shell = '\n      <iframe id="%HTML_ID%" width="100%" height="%HEIGHT%" frameborder="0">\n      </iframe>\n      <script>\n        (function() {\n          const frame = document.getElementById(%JSON_ID%);\n          const url = new URL(%URL%, window.location);\n          const port = %PORT%;\n          if (port) {\n            url.port = port;\n          }\n          frame.src = url;\n        })();\n      </script>\n    '
    proxy_url = os.environ.get('TENSORBOARD_PROXY_URL')
    if proxy_url is not None:
        proxy_url = proxy_url.replace('%PORT%', '%d' % port)
        replacements = [('%HTML_ID%', html.escape(frame_id, quote=True)), ('%JSON_ID%', json.dumps(frame_id)), ('%HEIGHT%', '%d' % height), ('%PORT%', '0'), ('%URL%', json.dumps(proxy_url))]
    else:
        replacements = [('%HTML_ID%', html.escape(frame_id, quote=True)), ('%JSON_ID%', json.dumps(frame_id)), ('%HEIGHT%', '%d' % height), ('%PORT%', '%d' % port), ('%URL%', json.dumps('/'))]
    for k, v in replacements:
        shell = shell.replace(k, v)
    iframe = IPython.display.HTML(shell)
    if display_handle:
        display_handle.update(iframe)
    else:
        IPython.display.display(iframe)