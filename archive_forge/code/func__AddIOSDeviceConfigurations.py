import copy
import gyp.common
import os
import os.path
import re
import shlex
import subprocess
import sys
from gyp.common import GypError
def _AddIOSDeviceConfigurations(targets):
    """Clone all targets and append -iphoneos to the name. Configure these targets
  to build for iOS devices and use correct architectures for those builds."""
    for target_dict in targets.values():
        toolset = target_dict['toolset']
        configs = target_dict['configurations']
        for config_name, simulator_config_dict in dict(configs).items():
            iphoneos_config_dict = copy.deepcopy(simulator_config_dict)
            configs[config_name + '-iphoneos'] = iphoneos_config_dict
            configs[config_name + '-iphonesimulator'] = simulator_config_dict
            if toolset == 'target':
                simulator_config_dict['xcode_settings']['SDKROOT'] = 'iphonesimulator'
                iphoneos_config_dict['xcode_settings']['SDKROOT'] = 'iphoneos'
    return targets