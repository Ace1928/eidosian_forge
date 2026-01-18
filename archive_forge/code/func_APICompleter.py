from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as c_exc
from googlecloudsdk.calliope import parser_extensions
from googlecloudsdk.command_lib.util.apis import arg_marshalling
from googlecloudsdk.command_lib.util.apis import registry
def APICompleter(**_):
    return [a.name for a in registry.GetAllAPIs()]