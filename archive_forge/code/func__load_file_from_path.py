import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_community.document_loaders.base import BaseLoader
def _load_file_from_path(self, file_path: str) -> Optional[Document]:
    """Load a file from a Dropbox path."""
    dbx = self._create_dropbox_client()
    try:
        from dropbox import exceptions
    except ImportError:
        raise ImportError('You must run `pip install dropbox')
    try:
        file_metadata = dbx.files_get_metadata(file_path)
        if file_metadata.is_downloadable:
            _, response = dbx.files_download(file_path)
        elif file_metadata.export_info:
            _, response = dbx.files_export(file_path, 'markdown')
    except exceptions.ApiError as ex:
        raise ValueError(f'Could not load file: {file_path}. Please verify the file pathand try again.') from ex
    try:
        text = response.content.decode('utf-8')
    except UnicodeDecodeError:
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.pdf':
            print(f'File {file_path} type detected as .pdf')
            from langchain_community.document_loaders import UnstructuredPDFLoader
            temp_dir = tempfile.TemporaryDirectory()
            temp_pdf = Path(temp_dir.name) / 'tmp.pdf'
            with open(temp_pdf, mode='wb') as f:
                f.write(response.content)
            try:
                loader = UnstructuredPDFLoader(str(temp_pdf))
                docs = loader.load()
                if docs:
                    return docs[0]
            except Exception as pdf_ex:
                print(f'Error while trying to parse PDF {file_path}: {pdf_ex}')
                return None
        else:
            print(f'File {file_path} could not be decoded as pdf or text. Skipping.')
        return None
    metadata = {'source': f'dropbox://{file_path}', 'title': os.path.basename(file_path)}
    return Document(page_content=text, metadata=metadata)