import abc
import contextlib
import hashlib
import os
from cinderclient.apiclient import base as common_base
from cinderclient import exceptions
from cinderclient import utils
def _build_list_url(self, resource_type, detailed=True, search_opts=None, marker=None, limit=None, sort=None, offset=None):
    if search_opts is None:
        search_opts = {}
    query_params = {}
    for key, val in search_opts.items():
        if val:
            query_params[key] = val
    if marker:
        query_params['marker'] = marker
    if limit:
        query_params['limit'] = limit
    if sort:
        query_params['sort'] = self._format_sort_param(sort, resource_type)
    if offset:
        query_params['offset'] = offset
    query_params = query_params
    query_string = utils.build_query_param(query_params, sort=True)
    detail = ''
    if detailed:
        detail = '/detail'
    return '/%(resource_type)s%(detail)s%(query_string)s' % {'resource_type': resource_type, 'detail': detail, 'query_string': query_string}