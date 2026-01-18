from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
class IBucketFilter(Interface):

    def getBucketFor(*somethings, **some_kw):
        """
        Return a L{Bucket} corresponding to the provided parameters.

        @returntype: L{Bucket}
        """