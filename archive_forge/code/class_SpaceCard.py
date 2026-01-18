import os
import re
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Type, Union
import requests
import yaml
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.hf_api import upload_file
from huggingface_hub.repocard_data import (
from huggingface_hub.utils import get_session, is_jinja_available, yaml_dump
from .constants import REPOCARD_NAME
from .utils import EntryNotFoundError, SoftTemporaryDirectory, logging, validate_hf_hub_args
class SpaceCard(RepoCard):
    card_data_class = SpaceCardData
    default_template_path = TEMPLATE_MODELCARD_PATH
    repo_type = 'space'