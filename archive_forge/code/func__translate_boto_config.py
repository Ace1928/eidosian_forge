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
def _translate_boto_config(self):
    """Translates boto config options to gcloud storage properties.

    Returns:
      A tuple where first element is a list of flags and the second element is
      a dict representing the env variables that can be set to set the
      gcloud storage properties.
    """
    flags = []
    env_vars = {}
    gcs_json_endpoint = _get_gcs_json_endpoint_from_boto_config(config)
    if gcs_json_endpoint:
        env_vars['CLOUDSDK_API_ENDPOINT_OVERRIDES_STORAGE'] = gcs_json_endpoint
    s3_endpoint = _get_s3_endpoint_from_boto_config(config)
    if s3_endpoint:
        env_vars['CLOUDSDK_STORAGE_S3_ENDPOINT_URL'] = s3_endpoint
    decryption_keys = []
    for section_name, section in config.items():
        for key, value in section.items():
            if key == 'encryption_key' and self.command_name in ENCRYPTION_SUPPORTED_COMMANDS:
                flags.append('--encryption-key=' + value)
            elif DECRYPTION_KEY_REGEX.match(key) and self.command_name in ENCRYPTION_SUPPORTED_COMMANDS:
                decryption_keys.append(value)
            elif key == 'content_language' and self.command_name in COMMANDS_SUPPORTING_ALL_HEADERS:
                flags.append('--content-language=' + value)
            elif key in _REQUIRED_BOTO_CONFIG_NOT_YET_SUPPORTED:
                self.logger.error('The boto config field {}:{} cannot be translated to gcloud storage equivalent.'.format(section_name, key))
            elif key == 'https_validate_certificates' and (not value):
                env_vars['CLOUDSDK_AUTH_DISABLE_SSL_VALIDATION'] = True
            elif key in ('gs_access_key_id', 'gs_secret_access_key') and (not boto_util.UsingGsHmac()):
                self.logger.debug('The boto config field {}:{} skipped translation to the gcloud storage equivalent as it would have been unused in gsutil.'.format(section_name, key))
            else:
                env_var = _BOTO_CONFIG_MAP.get(section_name, {}).get(key, None)
                if env_var is not None:
                    env_vars[env_var] = value
    if decryption_keys:
        flags.append('--decryption-keys=' + ','.join(decryption_keys))
    return (flags, env_vars)