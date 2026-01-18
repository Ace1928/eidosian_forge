import os
from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple, Union
from typing import List as LList
from urllib.parse import urlparse, urlunparse
from pydantic import ConfigDict, Field, validator
from pydantic.dataclasses import dataclass
import wandb
from . import expr_parsing, gql, internal
from .internal import (
def _url_to_report_id(url):
    parse_result = urlparse(url)
    path = parse_result.path
    _, entity, project, _, name = path.split('/')
    title, report_id = name.split('--')
    return report_id