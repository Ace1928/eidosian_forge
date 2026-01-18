from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import enum
def ControlPlaneNamespaceFromProductType(product_type):
    if product_type == Product.CLOUDRUN:
        return CLOUDRUN_EVENTS_NAMESPACE
    elif product_type == Product.KUBERUN:
        return KUBERUN_EVENTS_NAMESPACE
    else:
        raise ValueError('Invalid product_type found')