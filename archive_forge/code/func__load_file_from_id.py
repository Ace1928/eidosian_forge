import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator, validator
from langchain_community.document_loaders.base import BaseLoader
def _load_file_from_id(self, id: str) -> List[Document]:
    """Load a file from an ID."""
    from io import BytesIO
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    creds = self._load_credentials()
    service = build('drive', 'v3', credentials=creds)
    file = service.files().get(fileId=id, supportsAllDrives=True).execute()
    request = service.files().get_media(fileId=id)
    fh = BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    if self.file_loader_cls is not None:
        fh.seek(0)
        loader = self.file_loader_cls(file=fh, **self.file_loader_kwargs)
        docs = loader.load()
        for doc in docs:
            doc.metadata['source'] = f'https://drive.google.com/file/d/{id}/view'
            if 'title' not in doc.metadata:
                doc.metadata['title'] = f'{file.get('name')}'
        return docs
    else:
        from PyPDF2 import PdfReader
        content = fh.getvalue()
        pdf_reader = PdfReader(BytesIO(content))
        return [Document(page_content=page.extract_text(), metadata={'source': f'https://drive.google.com/file/d/{id}/view', 'title': f'{file.get('name')}', 'page': i}) for i, page in enumerate(pdf_reader.pages)]