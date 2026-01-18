from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING, Any, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities.vertexai import get_client_info
def _import_google_cloud_texttospeech() -> Any:
    try:
        from google.cloud import texttospeech
    except ImportError as e:
        raise ImportError('Cannot import google.cloud.texttospeech, please install `pip install google-cloud-texttospeech`.') from e
    return texttospeech