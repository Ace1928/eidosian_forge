import json
import zipfile
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
@staticmethod
def _get_channel_id_map(zip_path: Path) -> Dict[str, str]:
    """Get a dictionary mapping channel names to their respective IDs."""
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        try:
            with zip_file.open('channels.json', 'r') as f:
                channels = json.load(f)
            return {channel['name']: channel['id'] for channel in channels}
        except KeyError:
            return {}