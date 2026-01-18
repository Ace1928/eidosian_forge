from __future__ import annotations
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.pydantic_v1.dataclasses import dataclass
from langchain_community.document_loaders.base import BaseLoader
@dataclass
class GoogleApiClient:
    """Generic Google API Client.

    To use, you should have the ``google_auth_oauthlib,youtube_transcript_api,google``
    python package installed.
    As the google api expects credentials you need to set up a google account and
    register your Service. "https://developers.google.com/docs/api/quickstart/python"



    Example:
        .. code-block:: python

            from langchain_community.document_loaders import GoogleApiClient
            google_api_client = GoogleApiClient(
                service_account_path=Path("path_to_your_sec_file.json")
            )

    """
    credentials_path: Path = Path.home() / '.credentials' / 'credentials.json'
    service_account_path: Path = Path.home() / '.credentials' / 'credentials.json'
    token_path: Path = Path.home() / '.credentials' / 'token.json'

    def __post_init__(self) -> None:
        self.creds = self._load_credentials()

    @root_validator
    def validate_channel_or_videoIds_is_set(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that either folder_id or document_ids is set, but not both."""
        if not values.get('credentials_path') and (not values.get('service_account_path')):
            raise ValueError('Must specify either channel_name or video_ids')
        return values

    def _load_credentials(self) -> Any:
        """Load credentials."""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2 import service_account
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            raise ImportError('You must run`pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib youtube-transcript-api` to use the Google Drive loader')
        creds = None
        if self.service_account_path.exists():
            return service_account.Credentials.from_service_account_file(str(self.service_account_path))
        if self.token_path.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(str(self.credentials_path), SCOPES)
                creds = flow.run_local_server(port=0)
            with open(self.token_path, 'w') as token:
                token.write(creds.to_json())
        return creds