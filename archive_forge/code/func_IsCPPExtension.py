import gyp
import gyp.common
import gyp.generator.make as make  # Reuse global functions from make backend.
import os
import re
import subprocess
def IsCPPExtension(ext):
    return make.COMPILABLE_EXTENSIONS.get(ext) == 'cxx'