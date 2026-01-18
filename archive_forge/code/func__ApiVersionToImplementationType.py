import importlib
import os
import sys
import warnings
def _ApiVersionToImplementationType(api_version):
    if api_version == 2:
        return 'cpp'
    if api_version == 1:
        raise ValueError('api_version=1 is no longer supported.')
    if api_version == 0:
        return 'python'
    return None