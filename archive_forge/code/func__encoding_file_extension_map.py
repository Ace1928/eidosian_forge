from __future__ import annotations
import tempfile
from typing import TYPE_CHECKING, Any, Optional
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from langchain_community.utilities.vertexai import get_client_info
def _encoding_file_extension_map(encoding: texttospeech.AudioEncoding) -> Optional[str]:
    texttospeech = _import_google_cloud_texttospeech()
    ENCODING_FILE_EXTENSION_MAP = {texttospeech.AudioEncoding.LINEAR16: '.wav', texttospeech.AudioEncoding.MP3: '.mp3', texttospeech.AudioEncoding.OGG_OPUS: '.ogg', texttospeech.AudioEncoding.MULAW: '.wav', texttospeech.AudioEncoding.ALAW: '.wav'}
    return ENCODING_FILE_EXTENSION_MAP.get(encoding)