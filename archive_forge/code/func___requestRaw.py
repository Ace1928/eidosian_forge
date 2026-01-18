import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException
def __requestRaw(self, cnx: Optional[Union[HTTPRequestsConnectionClass, HTTPSRequestsConnectionClass]], verb: str, url: str, requestHeaders: Dict[str, str], input: Optional[Any]) -> Tuple[int, Dict[str, Any], str]:
    self.__deferRequest(verb)
    try:
        original_cnx = cnx
        if cnx is None:
            cnx = self.__createConnection()
        cnx.request(verb, url, input, requestHeaders)
        response = cnx.getresponse()
        status = response.status
        responseHeaders = {k.lower(): v for k, v in response.getheaders()}
        output = response.read()
        if input:
            if isinstance(input, IOBase):
                input.close()
        self.__log(verb, url, requestHeaders, input, status, responseHeaders, output)
        if status == 202 and (verb == 'GET' or verb == 'HEAD'):
            time.sleep(Consts.PROCESSING_202_WAIT_TIME)
            return self.__requestRaw(original_cnx, verb, url, requestHeaders, input)
        if status == 301 and 'location' in responseHeaders:
            location = responseHeaders['location']
            o = urllib.parse.urlparse(location)
            if o.scheme != self.__scheme:
                raise RuntimeError(f'Github server redirected from {self.__scheme} protocol to {o.scheme}, please correct your Github server URL via base_url: Github(base_url=...)')
            if o.hostname != self.__hostname:
                raise RuntimeError(f'Github server redirected from host {self.__hostname} to {o.hostname}, please correct your Github server URL via base_url: Github(base_url=...)')
            if o.path == url:
                port = ':' + str(self.__port) if self.__port is not None else ''
                requested_location = f'{self.__scheme}://{self.__hostname}{port}{url}'
                raise RuntimeError(f'Requested {requested_location} but server redirected to {location}, you may need to correct your Github server URL via base_url: Github(base_url=...)')
            if self._logger.isEnabledFor(logging.INFO):
                self._logger.info(f'Following Github server redirection from {url} to {o.path}')
            return self.__requestRaw(original_cnx, verb, o.path, requestHeaders, input)
        return (status, responseHeaders, output)
    finally:
        self.__recordRequestTime(verb)