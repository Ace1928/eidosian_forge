import ast
import gyp.common
import gyp.simple_copy
import multiprocessing
import os.path
import re
import shlex
import signal
import subprocess
import sys
import threading
import traceback
from distutils.version import StrictVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def SetUpConfigurations(target, target_dict):
    key_suffixes = ['=', '+', '?', '!', '/']
    build_file = gyp.common.BuildFile(target)
    if 'configurations' not in target_dict:
        target_dict['configurations'] = {'Default': {}}
    if 'default_configuration' not in target_dict:
        concrete = [i for i, config in target_dict['configurations'].items() if not config.get('abstract')]
        target_dict['default_configuration'] = sorted(concrete)[0]
    merged_configurations = {}
    configs = target_dict['configurations']
    for configuration, old_configuration_dict in configs.items():
        if old_configuration_dict.get('abstract'):
            continue
        new_configuration_dict = {}
        for key, target_val in target_dict.items():
            key_ext = key[-1:]
            if key_ext in key_suffixes:
                key_base = key[:-1]
            else:
                key_base = key
            if key_base not in non_configuration_keys:
                new_configuration_dict[key] = gyp.simple_copy.deepcopy(target_val)
        MergeConfigWithInheritance(new_configuration_dict, build_file, target_dict, configuration, [])
        merged_configurations[configuration] = new_configuration_dict
    for configuration in merged_configurations.keys():
        target_dict['configurations'][configuration] = merged_configurations[configuration]
    configs = target_dict['configurations']
    target_dict['configurations'] = {k: v for k, v in configs.items() if not v.get('abstract')}
    delete_keys = []
    for key in target_dict:
        key_ext = key[-1:]
        if key_ext in key_suffixes:
            key_base = key[:-1]
        else:
            key_base = key
        if key_base not in non_configuration_keys:
            delete_keys.append(key)
    for key in delete_keys:
        del target_dict[key]
    for configuration in target_dict['configurations'].keys():
        configuration_dict = target_dict['configurations'][configuration]
        for key in configuration_dict.keys():
            if key in invalid_configuration_keys:
                raise GypError('%s not allowed in the %s configuration, found in target %s' % (key, configuration, target))