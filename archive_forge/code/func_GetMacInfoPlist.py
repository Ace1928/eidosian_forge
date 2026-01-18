import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def GetMacInfoPlist(product_dir, xcode_settings, gyp_path_to_build_path):
    """Returns (info_plist, dest_plist, defines, extra_env), where:
  * |info_plist| is the source plist path, relative to the
    build directory,
  * |dest_plist| is the destination plist path, relative to the
    build directory,
  * |defines| is a list of preprocessor defines (empty if the plist
    shouldn't be preprocessed,
  * |extra_env| is a dict of env variables that should be exported when
    invoking |mac_tool copy-info-plist|.

  Only call this for mac bundle targets.

  Args:
      product_dir: Path to the directory containing the output bundle,
          relative to the build directory.
      xcode_settings: The XcodeSettings of the current target.
      gyp_to_build_path: A function that converts paths relative to the
          current gyp file to paths relative to the build directory.
  """
    info_plist = xcode_settings.GetPerTargetSetting('INFOPLIST_FILE')
    if not info_plist:
        return (None, None, [], {})
    assert ' ' not in info_plist, 'Spaces in Info.plist filenames not supported (%s)' % info_plist
    info_plist = gyp_path_to_build_path(info_plist)
    if xcode_settings.GetPerTargetSetting('INFOPLIST_PREPROCESS', default='NO') == 'YES':
        defines = shlex.split(xcode_settings.GetPerTargetSetting('INFOPLIST_PREPROCESSOR_DEFINITIONS', default=''))
    else:
        defines = []
    dest_plist = os.path.join(product_dir, xcode_settings.GetBundlePlistPath())
    extra_env = xcode_settings.GetPerTargetSettings()
    return (info_plist, dest_plist, defines, extra_env)