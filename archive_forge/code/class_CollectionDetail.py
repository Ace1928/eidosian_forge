from __future__ import annotations
import abc
import collections
import os
import typing as t
from ...util import (
from .. import (
class CollectionDetail:
    """Details about the layout of the current collection."""

    def __init__(self, name: str, namespace: str, root: str) -> None:
        self.name = name
        self.namespace = namespace
        self.root = root
        self.full_name = '%s.%s' % (namespace, name)
        self.prefix = '%s.' % self.full_name
        self.directory = os.path.join('ansible_collections', namespace, name)