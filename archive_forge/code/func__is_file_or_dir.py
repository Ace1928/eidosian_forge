from datetime import datetime
import mimetypes
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Union
from ..client import Client, register_client_class
from ..cloudpath import implementation_registry
from ..enums import FileCacheMode
from ..exceptions import MissingCredentialsError
from .azblobpath import AzureBlobPath
def _is_file_or_dir(self, cloud_path: AzureBlobPath) -> Optional[str]:
    if not cloud_path.blob:
        return 'dir'
    try:
        self._get_metadata(cloud_path)
        return 'file'
    except ResourceNotFoundError:
        prefix = cloud_path.blob
        if prefix and (not prefix.endswith('/')):
            prefix += '/'
        container_client = self.service_client.get_container_client(cloud_path.container)
        try:
            next(container_client.list_blobs(name_starts_with=prefix))
            return 'dir'
        except StopIteration:
            return None