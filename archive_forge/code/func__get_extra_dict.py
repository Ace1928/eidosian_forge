from libcloud.utils.py3 import u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.utils.misc import ReprMixin
from libcloud.common.types import LibcloudError
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.loadbalancer.base import Driver, Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
def _get_extra_dict(self, element, mapping):
    """
        Extract attributes from the element based on rules provided in the
        mapping dictionary.

        :param      element: Element to parse the values from.
        :type       element: xml.etree.ElementTree.Element.

        :param      mapping: Dictionary with the extra layout
        :type       node: :class:`Node`

        :rtype: ``dict``
        """
    extra = {}
    for attribute, values in mapping.items():
        transform_func = values['transform_func']
        value = findattr(element=element, xpath=values['xpath'], namespace=self.namespace)
        if value:
            try:
                extra[attribute] = transform_func(value)
            except Exception:
                extra[attribute] = None
        else:
            extra[attribute] = value
    return extra