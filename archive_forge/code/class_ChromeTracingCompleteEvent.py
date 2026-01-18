import os
import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import List, Dict, Union
import ray
@dataclass(init=True)
class ChromeTracingCompleteEvent:
    cat: str
    name: str
    pid: int
    tid: int
    ts: int
    dur: int
    cname: str
    args: Dict[str, Union[str, int]]
    ph: str = 'X'