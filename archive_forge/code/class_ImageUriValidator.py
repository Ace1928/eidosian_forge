import base64
import numbers
import textwrap
import uuid
from importlib import import_module
import copy
import io
import re
import sys
import warnings
from _plotly_utils.optional_imports import get_module
class ImageUriValidator(BaseValidator):
    _PIL = None
    try:
        _PIL = import_module('PIL')
    except ImportError:
        pass

    def __init__(self, plotly_name, parent_name, **kwargs):
        super(ImageUriValidator, self).__init__(plotly_name=plotly_name, parent_name=parent_name, **kwargs)

    def description(self):
        desc = "    The '{plotly_name}' property is an image URI that may be specified as:\n      - A remote image URI string\n        (e.g. 'http://www.somewhere.com/image.png')\n      - A data URI image string\n        (e.g. 'data:image/png;base64,iVBORw0KGgoAAAANSU')\n      - A PIL.Image.Image object which will be immediately converted\n        to a data URI image string\n        See http://pillow.readthedocs.io/en/latest/reference/Image.html\n        ".format(plotly_name=self.plotly_name)
        return desc

    def validate_coerce(self, v):
        if v is None:
            pass
        elif isinstance(v, str):
            pass
        elif self._PIL and isinstance(v, self._PIL.Image.Image):
            v = self.pil_image_to_uri(v)
        else:
            self.raise_invalid_val(v)
        return v

    @staticmethod
    def pil_image_to_uri(v):
        in_mem_file = io.BytesIO()
        v.save(in_mem_file, format='PNG')
        in_mem_file.seek(0)
        img_bytes = in_mem_file.read()
        base64_encoded_result_bytes = base64.b64encode(img_bytes)
        base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
        v = 'data:image/png;base64,{base64_encoded_result_str}'.format(base64_encoded_result_str=base64_encoded_result_str)
        return v