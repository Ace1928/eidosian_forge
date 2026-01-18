from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from enum import Enum
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
class Input(object):

    def __init__(self, command_type, data):
        self.confirmation_prompt = data.get('confirmation_prompt')
        self.default_continue = data.get('default_continue', True)
        if not self.confirmation_prompt and command_type is CommandType.DELETE:
            self.confirmation_prompt = 'You are about to delete {{{}}} [{{{}}}]'.format(util.RESOURCE_TYPE_FORMAT_KEY, util.NAME_FORMAT_KEY)