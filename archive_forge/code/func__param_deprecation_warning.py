from __future__ import absolute_import, division, print_function
from os import environ
from urllib.parse import urljoin
import platform
def _param_deprecation_warning(module, old_param, new_param, vers):
    if old_param in module.params:
        module.warn(f"{old_param} parameter is deprecated and will be removed in version {vers} Please use {new_param} parameter instead. Don't use both parameters simultaneously.")