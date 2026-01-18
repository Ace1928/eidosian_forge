from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from googlecloudsdk.core.console import console_attr as ca
from googlecloudsdk.core.resource import resource_printer_base
import six
Override to describe the format of the record.

    Takes in the raw record, returns a structure of "marker classes" (above in
    this file) that will describe how to print it.

    Args:
      record: The record to transform
    Returns:
      A structure of "marker classes" that describes how to print the record.
    