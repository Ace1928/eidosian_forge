import re
from typing import Optional, List, Union, Optional, Any, Dict
from dataclasses import dataclass
from lazyops import get_logger, PathIO
from lazyops.lazyclasses import lazyclass
@lazyclass
@dataclass
class TFSModelVersion:
    step: int
    label: str = 'latest'