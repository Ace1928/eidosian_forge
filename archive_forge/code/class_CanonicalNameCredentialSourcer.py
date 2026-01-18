import datetime
import getpass
import json
import logging
import os
import subprocess
import threading
import time
from collections import namedtuple
from copy import deepcopy
from hashlib import sha1
from dateutil.parser import parse
from dateutil.tz import tzlocal, tzutc
import botocore.compat
import botocore.configloader
from botocore import UNSIGNED
from botocore.compat import compat_shell_split, total_seconds
from botocore.config import Config
from botocore.exceptions import (
from botocore.tokens import SSOTokenProvider
from botocore.utils import (
class CanonicalNameCredentialSourcer:

    def __init__(self, providers):
        self._providers = providers

    def is_supported(self, source_name):
        """Validates a given source name.

        :type source_name: str
        :param source_name: The value of credential_source in the config
            file. This is the canonical name of the credential provider.

        :rtype: bool
        :returns: True if the credential provider is supported,
            False otherwise.
        """
        return source_name in [p.CANONICAL_NAME for p in self._providers]

    def source_credentials(self, source_name):
        """Loads source credentials based on the provided configuration.

        :type source_name: str
        :param source_name: The value of credential_source in the config
            file. This is the canonical name of the credential provider.

        :rtype: Credentials
        """
        source = self._get_provider(source_name)
        if isinstance(source, CredentialResolver):
            return source.load_credentials()
        return source.load()

    def _get_provider(self, canonical_name):
        """Return a credential provider by its canonical name.

        :type canonical_name: str
        :param canonical_name: The canonical name of the provider.

        :raises UnknownCredentialError: Raised if no
            credential provider by the provided name
            is found.
        """
        provider = self._get_provider_by_canonical_name(canonical_name)
        if canonical_name.lower() in ['sharedconfig', 'sharedcredentials']:
            assume_role_provider = self._get_provider_by_method('assume-role')
            if assume_role_provider is not None:
                if provider is None:
                    return assume_role_provider
                return CredentialResolver([assume_role_provider, provider])
        if provider is None:
            raise UnknownCredentialError(name=canonical_name)
        return provider

    def _get_provider_by_canonical_name(self, canonical_name):
        """Return a credential provider by its canonical name.

        This function is strict, it does not attempt to address
        compatibility issues.
        """
        for provider in self._providers:
            name = provider.CANONICAL_NAME
            if name and name.lower() == canonical_name.lower():
                return provider

    def _get_provider_by_method(self, method):
        """Return a credential provider by its METHOD name."""
        for provider in self._providers:
            if provider.METHOD == method:
                return provider