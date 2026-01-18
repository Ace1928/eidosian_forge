from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetJobURI(resource):
    job = resources.REGISTRY.ParseRelativeName(resource.name, collection='datapipelines.pipelines.jobs')
    return job.SelfLink()