import hashlib
import logging
from base64 import b64decode
from pathlib import Path
from time import strptime
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _parse_resource(resource: list) -> dict:
    rsc_dict: Dict[str, Any] = {}
    for elem in resource:
        if elem.tag == 'data':
            rsc_dict[elem.tag] = b64decode(elem.text) if elem.text else b''
            rsc_dict['hash'] = hashlib.md5(rsc_dict[elem.tag]).hexdigest()
        else:
            rsc_dict[elem.tag] = elem.text
    return rsc_dict