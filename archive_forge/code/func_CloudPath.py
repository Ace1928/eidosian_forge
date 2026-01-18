import abc
import mimetypes
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Generic, Callable, Iterable, Optional, Tuple, TypeVar, Union
from .cloudpath import CloudImplementation, CloudPath, implementation_registry
from .enums import FileCacheMode
from .exceptions import InvalidConfigurationException
def CloudPath(self, cloud_path: Union[str, BoundedCloudPath]) -> BoundedCloudPath:
    return self._cloud_meta.path_class(cloud_path=cloud_path, client=self)