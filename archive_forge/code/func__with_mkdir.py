import marshal
import pickle
from os import PathLike
from pathlib import Path
from typing import Union
from queuelib import queue
from scrapy.utils.request import request_from_dict
def _with_mkdir(queue_class):

    class DirectoriesCreated(queue_class):

        def __init__(self, path: Union[str, PathLike], *args, **kwargs):
            dirname = Path(path).parent
            if not dirname.exists():
                dirname.mkdir(parents=True, exist_ok=True)
            super().__init__(path, *args, **kwargs)
    return DirectoriesCreated