from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import enum
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import exceptions
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
import six
from six.moves import range  # pylint: disable=redefined-builtin
def ValidateInstanceInZone(instances, zone):
    """Validate if provided list in zone given as parameter.

  Args:
    instances: list of instances resources to be validated
    zone: a zone all instances must be in order to pass validation

  Raises:
    InvalidArgumentException: If any instance is in different zone
                              than given as parameter.
  """
    invalid_instances = [inst.SelfLink() for inst in instances if inst.zone != zone]
    if any(invalid_instances):
        raise calliope_exceptions.InvalidArgumentException('instances', 'The zone of instance must match the instance group zone. Following instances has invalid zone: %s' % ', '.join(invalid_instances))