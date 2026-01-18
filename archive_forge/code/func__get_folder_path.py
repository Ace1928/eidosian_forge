import contextlib
import re
from pathlib import Path
from typing import Any, List, Optional, Tuple
from urllib.parse import unquote
from langchain_core.documents import Document
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
def _get_folder_path(self, soup: Any) -> str:
    """Get the folder path to save the Documents in.

        Args:
            soup: BeautifulSoup4 soup object.

        Returns:
            Folder path.
        """
    course_name = soup.find('span', {'id': 'crumb_1'})
    if course_name is None:
        raise ValueError('No course name found.')
    course_name = course_name.text.strip()
    course_name_clean = unquote(course_name).replace(' ', '_').replace('/', '_').replace(':', '_').replace(',', '_').replace('?', '_').replace("'", '_').replace('!', '_').replace('"', '_')
    folder_path = Path('.') / course_name_clean
    return str(folder_path)