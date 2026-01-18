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
def _send_loader_doc(self, loading_end: bool=False) -> list:
    """Send documents fetched from loader to pebblo-server. Then send
        classified documents to Daxa cloud(If api_key is present). Internal method.

        Args:
            loading_end (bool, optional): Flag indicating the halt of data
                                        loading by loader. Defaults to False.
        """
    headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
    doc_content = [doc.dict() for doc in self.docs]
    docs = []
    for doc in doc_content:
        doc_authorized_identities = doc.get('metadata', {}).get('authorized_identities', [])
        doc_source_path = get_full_path(doc.get('metadata', {}).get('source', self.source_path))
        doc_source_owner = PebbloSafeLoader.get_file_owner_from_path(doc_source_path)
        doc_source_size = self.get_source_size(doc_source_path)
        page_content = str(doc.get('page_content'))
        page_content_size = self.calculate_content_size(page_content)
        self.source_aggr_size += page_content_size
        docs.append({'doc': page_content, 'source_path': doc_source_path, 'last_modified': doc.get('metadata', {}).get('last_modified'), 'file_owner': doc_source_owner, **({'authorized_identities': doc_authorized_identities} if doc_authorized_identities else {}), **({'source_path_size': doc_source_size} if doc_source_size is not None else {})})
    payload: Dict[str, Any] = {'name': self.app_name, 'owner': self.owner, 'docs': docs, 'plugin_version': PLUGIN_VERSION, 'load_id': self.load_id, 'loader_details': self.loader_details, 'loading_end': 'false', 'source_owner': self.source_owner}
    if loading_end is True:
        payload['loading_end'] = 'true'
        if 'loader_details' in payload:
            payload['loader_details']['source_aggr_size'] = self.source_aggr_size
    payload = Doc(**payload).dict(exclude_unset=True)
    load_doc_url = f'{CLASSIFIER_URL}{LOADER_DOC_URL}'
    classified_docs = []
    try:
        pebblo_resp = requests.post(load_doc_url, headers=headers, json=payload, timeout=300)
        classified_docs = json.loads(pebblo_resp.text).get('docs', None)
        if pebblo_resp.status_code not in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
            logger.warning('Received unexpected HTTP response code: %s', pebblo_resp.status_code)
        logger.debug('send_loader_doc[local]: request url %s, body %s len %s                    response status %s body %s', pebblo_resp.request.url, str(pebblo_resp.request.body), str(len(pebblo_resp.request.body if pebblo_resp.request.body else [])), str(pebblo_resp.status_code), pebblo_resp.json())
    except requests.exceptions.RequestException:
        logger.warning('Unable to reach pebblo server.')
    except Exception as e:
        logger.warning('An Exception caught in _send_loader_doc: %s', e)
    if self.api_key:
        if not classified_docs:
            logger.warning('No classified docs to send to pebblo-cloud.')
            return classified_docs
        try:
            payload['docs'] = classified_docs
            payload['classified'] = True
            headers.update({'x-api-key': self.api_key})
            pebblo_cloud_url = f'{PEBBLO_CLOUD_URL}{LOADER_DOC_URL}'
            pebblo_cloud_response = requests.post(pebblo_cloud_url, headers=headers, json=payload, timeout=20)
            logger.debug('send_loader_doc[cloud]: request url %s, body %s len %s                        response status %s body %s', pebblo_cloud_response.request.url, str(pebblo_cloud_response.request.body), str(len(pebblo_cloud_response.request.body if pebblo_cloud_response.request.body else [])), str(pebblo_cloud_response.status_code), pebblo_cloud_response.json())
        except requests.exceptions.RequestException:
            logger.warning('Unable to reach Pebblo cloud server.')
        except Exception as e:
            logger.warning('An Exception caught in _send_loader_doc: %s', e)
    if loading_end is True:
        PebbloSafeLoader.set_loader_sent()
    return classified_docs