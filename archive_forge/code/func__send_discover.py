import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional
import requests
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
def _send_discover(self) -> None:
    """Send app discovery payload to pebblo-server. Internal method."""
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    payload = self.app.dict(exclude_unset=True)
    app_discover_url = f'{CLASSIFIER_URL}{APP_DISCOVER_URL}'
    try:
        pebblo_resp = requests.post(app_discover_url, headers=headers, json=payload, timeout=20)
        logger.debug('send_discover[local]: request url %s, body %s len %s                    response status %s body %s', pebblo_resp.request.url, str(pebblo_resp.request.body), str(len(pebblo_resp.request.body if pebblo_resp.request.body else [])), str(pebblo_resp.status_code), pebblo_resp.json())
        if pebblo_resp.status_code in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
            PebbloSafeLoader.set_discover_sent()
        else:
            logger.warning(f'Received unexpected HTTP response code: {pebblo_resp.status_code}')
    except requests.exceptions.RequestException:
        logger.warning('Unable to reach pebblo server.')
    except Exception:
        logger.warning('An Exception caught in _send_discover.')
    if self.api_key:
        try:
            headers.update({'x-api-key': self.api_key})
            pebblo_cloud_url = f'{PEBBLO_CLOUD_URL}{APP_DISCOVER_URL}'
            pebblo_cloud_response = requests.post(pebblo_cloud_url, headers=headers, json=payload, timeout=20)
            logger.debug('send_discover[cloud]: request url %s, body %s len %s                        response status %s body %s', pebblo_cloud_response.request.url, str(pebblo_cloud_response.request.body), str(len(pebblo_cloud_response.request.body if pebblo_cloud_response.request.body else [])), str(pebblo_cloud_response.status_code), pebblo_cloud_response.json())
        except requests.exceptions.RequestException:
            logger.warning('Unable to reach Pebblo cloud server.')
        except Exception as e:
            logger.warning('An Exception caught in _send_discover: %s', e)