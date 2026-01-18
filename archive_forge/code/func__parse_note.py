import hashlib
import logging
from base64 import b64decode
from pathlib import Path
from time import strptime
from typing import Any, Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _parse_note(note: List, prefix: Optional[str]=None) -> dict:
    note_dict: Dict[str, Any] = {}
    resources = []

    def add_prefix(element_tag: str) -> str:
        if prefix is None:
            return element_tag
        return f'{prefix}.{element_tag}'
    for elem in note:
        if elem.tag == 'content':
            note_dict[elem.tag] = EverNoteLoader._parse_content(elem.text)
            note_dict['content-raw'] = elem.text
        elif elem.tag == 'resource':
            resources.append(EverNoteLoader._parse_resource(elem))
        elif elem.tag == 'created' or elem.tag == 'updated':
            note_dict[elem.tag] = strptime(elem.text, '%Y%m%dT%H%M%SZ')
        elif elem.tag == 'note-attributes':
            additional_attributes = EverNoteLoader._parse_note(elem, elem.tag)
            note_dict.update(additional_attributes)
        else:
            note_dict[elem.tag] = elem.text
    if len(resources) > 0:
        note_dict['resource'] = resources
    return {add_prefix(key): value for key, value in note_dict.items()}