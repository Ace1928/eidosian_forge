import logging
from botocore import xform_name
from .params import create_request_parameters
from .response import RawHandler, ResourceHandler
from .model import Action
from boto3.docs.docstring import ActionDocstring
from boto3.utils import inject_attribute
class WaiterAction(object):
    """
    A class representing a callable waiter action on a resource, for example
    ``s3.Bucket('foo').wait_until_bucket_exists()``.
    The waiter action may construct parameters from existing resource
    identifiers.

    :type waiter_model: :py:class`~boto3.resources.model.Waiter`
    :param waiter_model: The action waiter.
    :type waiter_resource_name: string
    :param waiter_resource_name: The name of the waiter action for the
                                 resource. It usually begins with a
                                 ``wait_until_``
    """

    def __init__(self, waiter_model, waiter_resource_name):
        self._waiter_model = waiter_model
        self._waiter_resource_name = waiter_resource_name

    def __call__(self, parent, *args, **kwargs):
        """
        Perform the wait operation after building operation
        parameters.

        :type parent: :py:class:`~boto3.resources.base.ServiceResource`
        :param parent: The resource instance to which this action is attached.
        """
        client_waiter_name = xform_name(self._waiter_model.waiter_name)
        params = create_request_parameters(parent, self._waiter_model)
        params.update(kwargs)
        logger.debug('Calling %s:%s with %r', parent.meta.service_name, self._waiter_resource_name, params)
        client = parent.meta.client
        waiter = client.get_waiter(client_waiter_name)
        response = waiter.wait(**params)
        logger.debug('Response: %r', response)