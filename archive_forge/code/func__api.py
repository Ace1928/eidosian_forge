from __future__ import absolute_import, division, print_function
import os
import sys
import copy
import json
import logging
import time
from datetime import datetime, timedelta
from ssl import SSLError
def _api(self, api_name, path, tenant, tenant_uuid, data=None, headers=None, timeout=None, api_version=None, **kwargs):
    """
        It calls the requests.Session APIs and handles session expiry
        and other situations where session needs to be reset.
        returns ApiResponse object
        :param path: takes relative path to the AVI api.
        :param tenant: overrides the tenant used during session creation
        :param tenant_uuid: overrides the tenant or tenant_uuid during session
            creation
        :param timeout: timeout for API calls; Default value is 60 seconds
        :param headers: dictionary of headers that override the session
            headers.
        """
    if self.pid != os.getpid():
        logger.info('pid %d change detected new %d. Closing session', self.pid, os.getpid())
        self.close()
        self.pid = os.getpid()
    if timeout is None:
        timeout = self.timeout
    fullpath = self._get_api_path(path)
    fn = getattr(super(ApiSession, self), api_name)
    api_hdrs = self._get_api_headers(tenant, tenant_uuid, timeout, headers, api_version)
    connection_error = False
    err = None
    cookies = {'csrftoken': api_hdrs['X-CSRFToken']}
    try:
        if self.session_cookie_name:
            cookies[self.session_cookie_name] = sessionDict[self.key]['session_id']
    except KeyError:
        pass
    try:
        if data is not None and type(data) is dict:
            resp = fn(fullpath, data=json.dumps(data), headers=api_hdrs, timeout=timeout, cookies=cookies, **kwargs)
        else:
            resp = fn(fullpath, data=data, headers=api_hdrs, timeout=timeout, cookies=cookies, **kwargs)
    except (RequestsConnectionError, SSLError) as e:
        logger.warning('Connection error retrying %s', e)
        if not self.retry_conxn_errors:
            raise
        connection_error = True
        err = e
    except Exception as e:
        logger.error('Error in Requests library %s', e)
        raise
    if not connection_error:
        logger.debug('path: %s http_method: %s hdrs: %s params: %s data: %s rsp: %s', fullpath, api_name.upper(), api_hdrs, kwargs, data, resp.text if self.data_log else 'None')
    if connection_error or resp.status_code in (401, 419):
        if connection_error:
            try:
                self.close()
            except Exception:
                pass
            logger.warning('Connection failed, retrying.')
            if self.retry_wait_time:
                time.sleep(self.retry_wait_time)
        else:
            logger.info('received error %d %s so resetting connection', resp.status_code, resp.text)
        ApiSession.reset_session(self)
        self.num_session_retries += 1
        if self.num_session_retries > self.max_session_retries:
            self.num_session_retries = 0
            if not connection_error:
                err = APIError('Status Code %s msg %s' % (resp.status_code, resp.text), resp)
            logger.error('giving up after %d retries conn failure %s err %s', self.max_session_retries, connection_error, err)
            ret_err = err if err else APIError('giving up after %d retries connection failure %s' % (self.max_session_retries, True))
            raise ret_err
        resp = self._api(api_name, path, tenant, tenant_uuid, data, headers=headers, api_version=api_version, timeout=timeout, **kwargs)
        self.num_session_retries = 0
    if resp.cookies and 'csrftoken' in resp.cookies:
        csrftoken = resp.cookies['csrftoken']
        self.headers.update({'X-CSRFToken': csrftoken})
    self._update_session_last_used()
    return ApiResponse.to_avi_response(resp)