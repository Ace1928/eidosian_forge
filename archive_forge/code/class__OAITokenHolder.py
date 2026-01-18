import json
import os
import time
from contextlib import contextmanager
from enum import Enum
from typing import NamedTuple, Optional
from unittest import mock
import requests
import mlflow
class _OAITokenHolder:

    def __init__(self, api_type):
        self._api_token = None
        self._credential = None
        self._is_azure_ad = api_type in ('azure_ad', 'azuread')
        self._key_configured = 'OPENAI_API_KEY' in os.environ
        if self._is_azure_ad and (not self._key_configured):
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError:
                raise mlflow.MlflowException('Using API type `azure_ad` or `azuread` requires the package `azure-identity` to be installed.')
            self._credential = DefaultAzureCredential()

    def validate(self, logger=None):
        """Validates the token or API key configured for accessing the OpenAI resource."""
        if self._key_configured:
            return
        if self._is_azure_ad:
            if not self._api_token or self._api_token.expires_on < time.time() + 60:
                from azure.core.exceptions import ClientAuthenticationError
                if logger:
                    logger.debug('Token for Azure AD is either expired or unset. Attempting to acquire a new token.')
                try:
                    self._api_token = self._credential.get_token('https://cognitiveservices.azure.com/.default')
                except ClientAuthenticationError as err:
                    raise mlflow.MlflowException(f'Unable to acquire a valid Azure AD token for the resource due to the following error: {err.message}') from err
                os.environ['OPENAI_API_KEY'] = self._api_token.token
            if logger:
                logger.debug('Token refreshed successfully')
        else:
            raise mlflow.MlflowException('OpenAI API key must be set in the ``OPENAI_API_KEY`` environment variable.')