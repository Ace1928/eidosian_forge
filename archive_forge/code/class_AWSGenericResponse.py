import hmac
import time
import base64
import hashlib
from typing import Dict, Type, Optional
from hashlib import sha256
from datetime import datetime
from libcloud.utils.py3 import ET, b, httplib, urlquote, basestring, _real_unicode
from libcloud.utils.xml import findall_ignore_namespace, findtext_ignore_namespace
from libcloud.common.base import BaseDriver, XmlResponse, JsonResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError, MalformedResponseError
class AWSGenericResponse(AWSBaseResponse):
    xpath = None
    exceptions = {}

    def success(self):
        return self.status in [httplib.OK, httplib.CREATED, httplib.ACCEPTED]

    def parse_error(self):
        context = self.connection.context
        status = int(self.status)
        if status == httplib.FORBIDDEN:
            if not self.body:
                raise InvalidCredsError(str(self.status) + ': ' + self.error)
            else:
                raise InvalidCredsError(self.body)
        try:
            body = ET.XML(self.body)
        except Exception:
            raise MalformedResponseError('Failed to parse XML', body=self.body, driver=self.connection.driver)
        if self.xpath:
            errs = findall_ignore_namespace(element=body, xpath=self.xpath, namespace=self.namespace)
        else:
            errs = [body]
        msgs = []
        for err in errs:
            code, message = self._parse_error_details(element=err)
            exceptionCls = self.exceptions.get(code, None)
            if exceptionCls is None:
                msgs.append('{}: {}'.format(code, message))
                continue
            params = {}
            if hasattr(exceptionCls, 'kwargs'):
                for key in exceptionCls.kwargs:
                    if key in context:
                        params[key] = context[key]
            raise exceptionCls(value=message, driver=self.connection.driver, **params)
        return '\n'.join(msgs)