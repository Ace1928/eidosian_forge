from ._base import *
from datetime import datetime
from typing import Optional, List, Union
from dataclasses import dataclass
from lazyops.lazyclasses.api import lazyclass
from lazyops.utils import timed_cache
@timed_cache(600)
def _init_newsplease():
    global _NP
    if _NP is not None:
        return
    nplease = lazy_init('news-please', 'newsplease')
    _NP = nplease.NewsPlease