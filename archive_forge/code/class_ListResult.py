from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import flags
from googlecloudsdk.command_lib.compute.tpus.execution_groups import util as tpu_utils
class ListResult(object):

    def __init__(self, name, status):
        self.name = name
        self.status = status

    def __gt__(self, lr):
        return self.name + self.status > lr.name + lr.status