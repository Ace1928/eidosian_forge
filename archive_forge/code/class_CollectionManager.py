import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
class CollectionManager(object):
    """
    A collection manager provides access to resource collection instances,
    which can be iterated and filtered. The manager exposes some
    convenience functions that are also found on resource collections,
    such as :py:meth:`~ResourceCollection.all` and
    :py:meth:`~ResourceCollection.filter`.

    Get all items::

        >>> for bucket in s3.buckets.all():
        ...     print(bucket.name)

    Get only some items via filtering::

        >>> for queue in sqs.queues.filter(QueueNamePrefix='AWS'):
        ...     print(queue.url)

    Get whole pages of items:

        >>> for page in s3.Bucket('boto3').objects.pages():
        ...     for obj in page:
        ...         print(obj.key)

    A collection manager is not iterable. You **must** call one of the
    methods that return a :py:class:`ResourceCollection` before trying
    to iterate, slice, or convert to a list.

    See the :ref:`guide_collections` guide for a high-level overview
    of collections, including when remote service requests are performed.

    :type collection_model: :py:class:`~boto3.resources.model.Collection`
    :param model: Collection model

    :type parent: :py:class:`~boto3.resources.base.ServiceResource`
    :param parent: The collection's parent resource

    :type factory: :py:class:`~boto3.resources.factory.ResourceFactory`
    :param factory: The resource factory to create new resources

    :type service_context: :py:class:`~boto3.utils.ServiceContext`
    :param service_context: Context about the AWS service
    """
    _collection_cls = ResourceCollection

    def __init__(self, collection_model, parent, factory, service_context):
        self._model = collection_model
        operation_name = self._model.request.operation
        self._parent = parent
        search_path = collection_model.resource.path
        self._handler = ResourceHandler(search_path=search_path, factory=factory, resource_model=collection_model.resource, service_context=service_context, operation_name=operation_name)

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._parent, '{0}.{1}'.format(self._parent.meta.service_name, self._model.resource.type))

    def iterator(self, **kwargs):
        """
        Get a resource collection iterator from this manager.

        :rtype: :py:class:`ResourceCollection`
        :return: An iterable representing the collection of resources
        """
        return self._collection_cls(self._model, self._parent, self._handler, **kwargs)

    def all(self):
        return self.iterator()
    all.__doc__ = ResourceCollection.all.__doc__

    def filter(self, **kwargs):
        return self.iterator(**kwargs)
    filter.__doc__ = ResourceCollection.filter.__doc__

    def limit(self, count):
        return self.iterator(limit=count)
    limit.__doc__ = ResourceCollection.limit.__doc__

    def page_size(self, count):
        return self.iterator(page_size=count)
    page_size.__doc__ = ResourceCollection.page_size.__doc__

    def pages(self):
        return self.iterator().pages()
    pages.__doc__ = ResourceCollection.pages.__doc__