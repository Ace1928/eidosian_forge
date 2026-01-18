from __future__ import absolute_import
import sys
import os
import re
from six import iteritems
from ..api_client import ApiClient

        replace status of the specified APIService
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please pass async_req=True
        >>> thread = api.replace_api_service_status_with_http_info(name, body,
        async_req=True)
        >>> result = thread.get()

        :param async_req bool
        :param str name: name of the APIService (required)
        :param V1beta1APIService body: (required)
        :param str pretty: If 'true', then the output is pretty printed.
        :param str dry_run: When present, indicates that modifications should
        not be persisted. An invalid or unrecognized dryRun directive will
        result in an error response and no further processing of the request.
        Valid values are: - All: all dry run stages will be processed
        :param str field_manager: fieldManager is a name associated with the
        actor or entity that is making these changes. The value must be less
        than or 128 characters long, and only contain printable characters, as
        defined by https://golang.org/pkg/unicode/#IsPrint.
        :return: V1beta1APIService
                 If the method is called asynchronously,
                 returns the request thread.
        