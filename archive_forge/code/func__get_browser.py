import os
import time
import numpy as np
import PyChromeDevTools
import ipyvolume as ipv
def _get_browser():
    options = []
    executable = os.environ.get('IPYVOLUME_HEADLESS_BROWSER')
    if executable:
        options.append(executable)
    options.append('/Applications/Google Chrome Canary.app/Contents/MacOS/Google Chrome Canary')
    options.append('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    for path in options:
        if os.path.exists(path):
            return path
    raise ValueError('no browser found, try setting the IPYVOLUME_HEADLESS_BROWSER environmental variable')