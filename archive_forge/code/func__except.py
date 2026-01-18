from pprint import pformat
from six import iteritems
import re
@_except.setter
def _except(self, _except):
    """
        Sets the _except of this V1beta1IPBlock.
        Except is a slice of CIDRs that should not be included within an IP
        Block Valid examples are "192.168.1.1/24" Except values will be
        rejected if they are outside the CIDR range

        :param _except: The _except of this V1beta1IPBlock.
        :type: list[str]
        """
    self.__except = _except