from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def GetAnnotationParentTemplate(project, location, asset_type, asset, annotation_set):
    return ANNOTATION_PARENT_TEMPLATE.format(project, location, asset_type, asset, annotation_set)