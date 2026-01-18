import fcntl
import fnmatch
import glob
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
import tempfile
def ExecCompileXcassets(self, keys, *inputs):
    """Compiles multiple .xcassets files into a single .car file.

    This invokes 'actool' to compile all the inputs .xcassets files. The
    |keys| arguments is a json-encoded dictionary of extra arguments to
    pass to 'actool' when the asset catalogs contains an application icon
    or a launch image.

    Note that 'actool' does not create the Assets.car file if the asset
    catalogs does not contains imageset.
    """
    command_line = ['xcrun', 'actool', '--output-format', 'human-readable-text', '--compress-pngs', '--notices', '--warnings', '--errors']
    is_iphone_target = 'IPHONEOS_DEPLOYMENT_TARGET' in os.environ
    if is_iphone_target:
        platform = os.environ['CONFIGURATION'].split('-')[-1]
        if platform not in ('iphoneos', 'iphonesimulator'):
            platform = 'iphonesimulator'
        command_line.extend(['--platform', platform, '--target-device', 'iphone', '--target-device', 'ipad', '--minimum-deployment-target', os.environ['IPHONEOS_DEPLOYMENT_TARGET'], '--compile', os.path.abspath(os.environ['CONTENTS_FOLDER_PATH'])])
    else:
        command_line.extend(['--platform', 'macosx', '--target-device', 'mac', '--minimum-deployment-target', os.environ['MACOSX_DEPLOYMENT_TARGET'], '--compile', os.path.abspath(os.environ['UNLOCALIZED_RESOURCES_FOLDER_PATH'])])
    if keys:
        keys = json.loads(keys)
        for key, value in keys.items():
            arg_name = '--' + key
            if isinstance(value, bool):
                if value:
                    command_line.append(arg_name)
            elif isinstance(value, list):
                for v in value:
                    command_line.append(arg_name)
                    command_line.append(str(v))
            else:
                command_line.append(arg_name)
                command_line.append(str(value))
    command_line.extend(map(os.path.abspath, inputs))
    subprocess.check_call(command_line)