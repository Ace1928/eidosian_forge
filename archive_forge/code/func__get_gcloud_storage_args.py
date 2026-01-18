from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
import os
import re
import subprocess
from boto import config
from gslib import exception
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
def _get_gcloud_storage_args(self, sub_opts, gsutil_args, gcloud_storage_map):
    if gcloud_storage_map is None:
        raise exception.GcloudStorageTranslationError('Command "{}" cannot be translated to gcloud storage because the translation mapping is missing.'.format(self.command_name))
    args = []
    if isinstance(gcloud_storage_map.gcloud_command, list):
        args.extend(gcloud_storage_map.gcloud_command)
    elif isinstance(gcloud_storage_map.gcloud_command, dict):
        if gcloud_storage_map.flag_map:
            raise ValueError('Flags mapping should not be present at the top-level command if a sub-command is used. Command: {}.'.format(self.command_name))
        sub_command = gsutil_args[0]
        sub_opts, parsed_args = self.ParseSubOpts(args=gsutil_args[1:], should_update_sub_opts_and_args=False)
        return self._get_gcloud_storage_args(sub_opts, parsed_args, gcloud_storage_map.gcloud_command.get(sub_command))
    else:
        raise ValueError('Incorrect mapping found for "{}" command'.format(self.command_name))
    if sub_opts:
        for option, value in sub_opts:
            if option not in gcloud_storage_map.flag_map:
                raise exception.GcloudStorageTranslationError('Command option "{}" cannot be translated to gcloud storage'.format(option))
            else:
                args.append(option)
                if value != '':
                    args.append(value)
    return _convert_args_to_gcloud_values(args + gsutil_args, gcloud_storage_map)