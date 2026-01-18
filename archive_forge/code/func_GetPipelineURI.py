from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetPipelineURI(resource):
    pipeline = resources.REGISTRY.ParseRelativeName(resource.name, collection='datapipelines.pipelines')
    return pipeline.SelfLink()