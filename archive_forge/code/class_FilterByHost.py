from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class FilterByHost(HierarchicalBucketFilter):
    """
    A Hierarchical Bucket filter with a L{Bucket} for each host.
    """
    sweepInterval = 60 * 20

    def getBucketKey(self, transport):
        return transport.getPeer()[1]