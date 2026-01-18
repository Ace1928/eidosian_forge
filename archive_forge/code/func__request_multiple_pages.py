import time
from libcloud.utils.py3 import _real_unicode as u
from libcloud.utils.xml import findall, findattr, findtext
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.common.aliyun import AliyunXmlResponse, SignedAliyunConnection
from libcloud.compute.types import NodeState, StorageVolumeState, VolumeSnapshotState
def _request_multiple_pages(self, path, params, parse_func):
    """
        Request all resources by multiple pages.
        :param path: the resource path
        :type path: ``str``
        :param params: the query parameters
        :type params: ``dict``
        :param parse_func: the function object to parse the response body
        :param type: ``function``
        :return: list of resource object, if not found any, return []
        :rtype: ``list``
        """
    results = []
    while True:
        one_page = self.connection.request(path, params).object
        resources = parse_func(one_page)
        results += resources
        pagination = self._get_pagination(one_page)
        if pagination.next() is None:
            break
        params.update(pagination.to_dict())
    return results