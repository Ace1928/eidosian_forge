import collections
import os
import re
import subprocess
import sys
from gyp.common import OrderedSet
import gyp.MSVSUtil
import gyp.MSVSVersion
def GetVSMacroEnv(self, base_to_build=None, config=None):
    """Get a dict of variables mapping internal VS macro names to their gyp
        equivalents."""
    target_arch = self.GetArch(config)
    if target_arch == 'x86':
        target_platform = 'Win32'
    else:
        target_platform = target_arch
    target_name = self.spec.get('product_prefix', '') + self.spec.get('product_name', self.spec['target_name'])
    target_dir = base_to_build + '\\' if base_to_build else ''
    target_ext = '.' + self.GetExtension()
    target_file_name = target_name + target_ext
    replacements = {'$(InputName)': '${root}', '$(InputPath)': '${source}', '$(IntDir)': '$!INTERMEDIATE_DIR', '$(OutDir)\\': target_dir, '$(PlatformName)': target_platform, '$(ProjectDir)\\': '', '$(ProjectName)': self.spec['target_name'], '$(TargetDir)\\': target_dir, '$(TargetExt)': target_ext, '$(TargetFileName)': target_file_name, '$(TargetName)': target_name, '$(TargetPath)': os.path.join(target_dir, target_file_name)}
    replacements.update(GetGlobalVSMacroEnv(self.vs_version))
    return replacements