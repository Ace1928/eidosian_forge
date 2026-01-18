from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from collections.abc import Container, Mapping
from googlecloudsdk.core import exceptions
def _GetItemSortKey(target):
    """Key function for sorting TrafficTarget objects during __getitem__."""
    percent = target.percent if target.percent else 0
    tag = target.tag if target.tag else ''
    return (percent, tag)