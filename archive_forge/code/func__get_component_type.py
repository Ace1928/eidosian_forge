from __future__ import annotations
import concurrent.futures
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
import urllib.parse
import uuid
import warnings
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Literal
import httpx
import huggingface_hub
from huggingface_hub import CommitOperationAdd, SpaceHardware, SpaceStage
from huggingface_hub.utils import (
from packaging import version
from gradio_client import utils
from gradio_client.compatibility import EndpointV3Compatibility
from gradio_client.data_classes import ParameterInfo
from gradio_client.documentation import document
from gradio_client.exceptions import AuthenticationError
from gradio_client.utils import (
def _get_component_type(self, component_id: int):
    component = next((i for i in self.client.config['components'] if i['id'] == component_id))
    skip_api = component.get('skip_api', component['type'] in utils.SKIP_COMPONENTS)
    return ComponentApiType(skip_api, self.value_is_file(component), component['type'] == 'state')