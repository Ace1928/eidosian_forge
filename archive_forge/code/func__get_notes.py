import json
import urllib
from datetime import datetime
from typing import Iterator, List, Optional
from langchain_core.documents import Document
from langchain_core.utils import get_from_env
from langchain_community.document_loaders.base import BaseLoader
def _get_notes(self) -> Iterator[Document]:
    has_more = True
    page = 1
    while has_more:
        req_note = urllib.request.Request(self._get_note_url.format(page=page))
        with urllib.request.urlopen(req_note) as response:
            json_data = json.loads(response.read().decode())
            for note in json_data['items']:
                metadata = {'source': LINK_NOTE_TEMPLATE.format(id=note['id']), 'folder': self._get_folder(note['parent_id']), 'tags': self._get_tags(note['id']), 'title': note['title'], 'created_time': self._convert_date(note['created_time']), 'updated_time': self._convert_date(note['updated_time'])}
                yield Document(page_content=note['body'], metadata=metadata)
            has_more = json_data['has_more']
            page += 1