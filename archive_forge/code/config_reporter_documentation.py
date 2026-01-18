from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
Make a value to insert into the GenerateConfigReport request.

    Args:
      value_type: The type to encode the message into. Generally, either
        OldConfigValue or NewConfigValue.

    Returns:
      The encoded config value object of type value_type.
    