import collections
import copy
import hashlib
import json
import multiprocessing
import os.path
import re
import signal
import subprocess
import sys
import gyp
import gyp.common
import gyp.msvs_emulation
import gyp.MSVSUtil as MSVSUtil
import gyp.xcode_emulation
from io import StringIO
from gyp.common import GetEnvironFallback
import gyp.ninja_syntax as ninja_syntax
def WriteMacXCassets(self, xcassets, bundle_depends):
    """Writes ninja edges for 'mac_bundle_resources' .xcassets files.

        This add an invocation of 'actool' via the 'mac_tool.py' helper script.
        It assumes that the assets catalogs define at least one imageset and
        thus an Assets.car file will be generated in the application resources
        directory. If this is not the case, then the build will probably be done
        at each invocation of ninja."""
    if not xcassets:
        return
    extra_arguments = {}
    settings_to_arg = {'XCASSETS_APP_ICON': 'app-icon', 'XCASSETS_LAUNCH_IMAGE': 'launch-image'}
    settings = self.xcode_settings.xcode_settings[self.config_name]
    for settings_key, arg_name in settings_to_arg.items():
        value = settings.get(settings_key)
        if value:
            extra_arguments[arg_name] = value
    partial_info_plist = None
    if extra_arguments:
        partial_info_plist = self.GypPathToUniqueOutput('assetcatalog_generated_info.plist')
        extra_arguments['output-partial-info-plist'] = partial_info_plist
    outputs = []
    outputs.append(os.path.join(self.xcode_settings.GetBundleResourceFolder(), 'Assets.car'))
    if partial_info_plist:
        outputs.append(partial_info_plist)
    keys = QuoteShellArgument(json.dumps(extra_arguments), self.flavor)
    extra_env = self.xcode_settings.GetPerTargetSettings()
    env = self.GetSortedXcodeEnv(additional_settings=extra_env)
    env = self.ComputeExportEnvString(env)
    bundle_depends.extend(self.ninja.build(outputs, 'compile_xcassets', xcassets, variables=[('env', env), ('keys', keys)]))
    return partial_info_plist