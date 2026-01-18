import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def CalculateCommonVariables(default_variables, params):
    generator_flags = params.get('generator_flags', {})
    msvs_version = gyp.msvs_emulation.GetVSVersion(generator_flags)
    default_variables['MSVS_VERSION'] = msvs_version.ShortName()
    if '64' in os.environ.get('PROCESSOR_ARCHITECTURE', '') or '64' in os.environ.get('PROCESSOR_ARCHITEW6432', ''):
        default_variables['MSVS_OS_BITS'] = 64
    else:
        default_variables['MSVS_OS_BITS'] = 32