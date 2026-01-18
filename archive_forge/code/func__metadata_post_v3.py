from datetime import datetime
import logging
import os
from typing import (
import warnings
import numpy as np
from ..core.request import Request, IOMode, InitializationError
from ..core.v3_plugin_api import PluginV3, ImageProperties
def _metadata_post_v3(self) -> Dict[str, Any]:
    """Extract XML metadata from SPE v3 files

        Returns
        -------
        dict with key `"__xml"`, whose value is the XML metadata
        """
    info = self._parse_header(Spec.basic, 'latin1')
    self._file.seek(info['xml_footer_offset'])
    xml = self._file.read()
    return {'__xml': xml}