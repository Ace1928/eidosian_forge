import os
import warnings
import xml.dom.minidom
from .base import (
def _gen_filename_from_param(self, param):
    base = param.getElementsByTagName('name')[0].firstChild.nodeValue
    fileExtensions = param.getAttribute('fileExtensions')
    if fileExtensions:
        ext = fileExtensions
    else:
        ext = {'image': '.nii', 'transform': '.txt', 'file': ''}[param.nodeName]
    return base + ext