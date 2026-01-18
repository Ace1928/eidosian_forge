from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class FilterByServer(HierarchicalBucketFilter):
    """
    A Hierarchical Bucket filter with a L{Bucket} for each service.
    """
    sweepInterval = None

    def getBucketKey(self, transport):
        return transport.getHost()[2]