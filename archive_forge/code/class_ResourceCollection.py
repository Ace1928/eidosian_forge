import copy
import logging
from botocore import xform_name
from botocore.utils import merge_dicts
from .action import BatchAction
from .params import create_request_parameters
from .response import ResourceHandler
from ..docs import docstring
class ResourceCollection(object):
    """
    Represents a collection of resources, which can be iterated through,
    optionally with filtering. Collections automatically handle pagination
    for you.

    See :ref:`guide_collections` for a high-level overview of collections,
    including when remote service requests are performed.

    :type model: :py:class:`~boto3.resources.model.Collection`
    :param model: Collection model
    :type parent: :py:class:`~boto3.resources.base.ServiceResource`
    :param parent: The collection's parent resource
    :type handler: :py:class:`~boto3.resources.response.ResourceHandler`
    :param handler: The resource response handler used to create resource
                    instances
    """

    def __init__(self, model, parent, handler, **kwargs):
        self._model = model
        self._parent = parent
        self._py_operation_name = xform_name(model.request.operation)
        self._handler = handler
        self._params = copy.deepcopy(kwargs)

    def __repr__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self._parent, '{0}.{1}'.format(self._parent.meta.service_name, self._model.resource.type))

    def __iter__(self):
        """
        A generator which yields resource instances after doing the
        appropriate service operation calls and handling any pagination
        on your behalf.

        Page size, item limit, and filter parameters are applied
        if they have previously been set.

            >>> bucket = s3.Bucket('boto3')
            >>> for obj in bucket.objects.all():
            ...     print(obj.key)
            'key1'
            'key2'

        """
        limit = self._params.get('limit', None)
        count = 0
        for page in self.pages():
            for item in page:
                yield item
                count += 1
                if limit is not None and count >= limit:
                    return

    def _clone(self, **kwargs):
        """
        Create a clone of this collection. This is used by the methods
        below to provide a chainable interface that returns copies
        rather than the original. This allows things like:

            >>> base = collection.filter(Param1=1)
            >>> query1 = base.filter(Param2=2)
            >>> query2 = base.filter(Param3=3)
            >>> query1.params
            {'Param1': 1, 'Param2': 2}
            >>> query2.params
            {'Param1': 1, 'Param3': 3}

        :rtype: :py:class:`ResourceCollection`
        :return: A clone of this resource collection
        """
        params = copy.deepcopy(self._params)
        merge_dicts(params, kwargs, append_lists=True)
        clone = self.__class__(self._model, self._parent, self._handler, **params)
        return clone

    def pages(self):
        """
        A generator which yields pages of resource instances after
        doing the appropriate service operation calls and handling
        any pagination on your behalf. Non-paginated calls will
        return a single page of items.

        Page size, item limit, and filter parameters are applied
        if they have previously been set.

            >>> bucket = s3.Bucket('boto3')
            >>> for page in bucket.objects.pages():
            ...     for obj in page:
            ...         print(obj.key)
            'key1'
            'key2'

        :rtype: list(:py:class:`~boto3.resources.base.ServiceResource`)
        :return: List of resource instances
        """
        client = self._parent.meta.client
        cleaned_params = self._params.copy()
        limit = cleaned_params.pop('limit', None)
        page_size = cleaned_params.pop('page_size', None)
        params = create_request_parameters(self._parent, self._model.request)
        merge_dicts(params, cleaned_params, append_lists=True)
        if client.can_paginate(self._py_operation_name):
            logger.debug('Calling paginated %s:%s with %r', self._parent.meta.service_name, self._py_operation_name, params)
            paginator = client.get_paginator(self._py_operation_name)
            pages = paginator.paginate(PaginationConfig={'MaxItems': limit, 'PageSize': page_size}, **params)
        else:
            logger.debug('Calling %s:%s with %r', self._parent.meta.service_name, self._py_operation_name, params)
            pages = [getattr(client, self._py_operation_name)(**params)]
        count = 0
        for page in pages:
            page_items = []
            for item in self._handler(self._parent, params, page):
                page_items.append(item)
                count += 1
                if limit is not None and count >= limit:
                    break
            yield page_items
            if limit is not None and count >= limit:
                break

    def all(self):
        """
        Get all items from the collection, optionally with a custom
        page size and item count limit.

        This method returns an iterable generator which yields
        individual resource instances. Example use::

            # Iterate through items
            >>> for queue in sqs.queues.all():
            ...     print(queue.url)
            'https://url1'
            'https://url2'

            # Convert to list
            >>> queues = list(sqs.queues.all())
            >>> len(queues)
            2
        """
        return self._clone()

    def filter(self, **kwargs):
        """
        Get items from the collection, passing keyword arguments along
        as parameters to the underlying service operation, which are
        typically used to filter the results.

        This method returns an iterable generator which yields
        individual resource instances. Example use::

            # Iterate through items
            >>> for queue in sqs.queues.filter(Param='foo'):
            ...     print(queue.url)
            'https://url1'
            'https://url2'

            # Convert to list
            >>> queues = list(sqs.queues.filter(Param='foo'))
            >>> len(queues)
            2

        :rtype: :py:class:`ResourceCollection`
        """
        return self._clone(**kwargs)

    def limit(self, count):
        """
        Return at most this many resources.

            >>> for bucket in s3.buckets.limit(5):
            ...     print(bucket.name)
            'bucket1'
            'bucket2'
            'bucket3'
            'bucket4'
            'bucket5'

        :type count: int
        :param count: Return no more than this many items
        :rtype: :py:class:`ResourceCollection`
        """
        return self._clone(limit=count)

    def page_size(self, count):
        """
        Fetch at most this many resources per service request.

            >>> for obj in s3.Bucket('boto3').objects.page_size(100):
            ...     print(obj.key)

        :type count: int
        :param count: Fetch this many items per request
        :rtype: :py:class:`ResourceCollection`
        """
        return self._clone(page_size=count)