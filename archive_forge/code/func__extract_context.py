import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
def _extract_context(self, compute):
    """
        Extract size, or node type, from a compute node XML representation.

        Extract node size, or node type, description from a compute node XML
        representation, converting the node size to a NodeSize object.

        :type  compute: :class:`ElementTree`
        :param compute: XML representation of a compute node.

        :rtype:  ``dict``
        :return: Dictionary containing (key, value) pairs related to
                 compute node context.
        """
    contexts = dict()
    context = compute.find('CONTEXT')
    if context is not None:
        for context_element in list(context):
            contexts[context_element.tag.lower()] = context_element.text
    return contexts