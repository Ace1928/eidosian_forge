from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.command_lib.redis import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.console import console_io
from six.moves import filter  # pylint: disable=redefined-builtin
def AddNewRedisConfigs(instance_ref, redis_configs_dict, patch_request):
    messages = util.GetMessagesForResource(instance_ref)
    new_redis_configs = util.PackageInstanceRedisConfig(redis_configs_dict, messages)
    patch_request.instance.redisConfigs = new_redis_configs
    patch_request = AddFieldToUpdateMask('redis_configs', patch_request)
    return patch_request