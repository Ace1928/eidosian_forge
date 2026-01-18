import os
import subprocess
import sys
import re
from distutils.errors import DistutilsExecError, DistutilsPlatformError, \
from distutils.ccompiler import CCompiler, gen_lib_options
from distutils import log
from distutils.util import get_platform
import winreg
def _remove_visual_c_ref(self, manifest_file):
    try:
        manifest_f = open(manifest_file)
        try:
            manifest_buf = manifest_f.read()
        finally:
            manifest_f.close()
        pattern = re.compile('<assemblyIdentity.*?name=("|\')Microsoft\\.VC\\d{2}\\.CRT("|\').*?(/>|</assemblyIdentity>)', re.DOTALL)
        manifest_buf = re.sub(pattern, '', manifest_buf)
        pattern = '<dependentAssembly>\\s*</dependentAssembly>'
        manifest_buf = re.sub(pattern, '', manifest_buf)
        pattern = re.compile('<assemblyIdentity.*?name=(?:"|\')(.+?)(?:"|\').*?(?:/>|</assemblyIdentity>)', re.DOTALL)
        if re.search(pattern, manifest_buf) is None:
            return None
        manifest_f = open(manifest_file, 'w')
        try:
            manifest_f.write(manifest_buf)
            return manifest_file
        finally:
            manifest_f.close()
    except OSError:
        pass