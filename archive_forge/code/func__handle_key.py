import asyncio
import datetime
from io import BytesIO, StringIO
import json
import logging
import os
from pathlib import Path
import numpy as np
from PIL import Image
from matplotlib import _api, backend_bases, backend_tools
from matplotlib.backends import backend_agg
from matplotlib.backend_bases import (
def _handle_key(self, event):
    KeyEvent(event['type'] + '_event', self, _handle_key(event['key']), *self._last_mouse_xy, guiEvent=event.get('guiEvent'))._process()