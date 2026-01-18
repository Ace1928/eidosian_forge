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
class AssumeRoleProvider(CredentialProvider):
    METHOD = 'assume-role'
    CANONICAL_NAME = None
    ROLE_CONFIG_VAR = 'role_arn'
    WEB_IDENTITY_TOKE_FILE_VAR = 'web_identity_token_file'
    EXPIRY_WINDOW_SECONDS = 60 * 15

    def __init__(self, load_config, client_creator, cache, profile_name, prompter=getpass.getpass, credential_sourcer=None, profile_provider_builder=None):
        """
        :type load_config: callable
        :param load_config: A function that accepts no arguments, and
            when called, will return the full configuration dictionary
            for the session (``session.full_config``).

        :type client_creator: callable
        :param client_creator: A factory function that will create
            a client when called.  Has the same interface as
            ``botocore.session.Session.create_client``.

        :type cache: dict
        :param cache: An object that supports ``__getitem__``,
            ``__setitem__``, and ``__contains__``.  An example
            of this is the ``JSONFileCache`` class in the CLI.

        :type profile_name: str
        :param profile_name: The name of the profile.

        :type prompter: callable
        :param prompter: A callable that returns input provided
            by the user (i.e raw_input, getpass.getpass, etc.).

        :type credential_sourcer: CanonicalNameCredentialSourcer
        :param credential_sourcer: A credential provider that takes a
            configuration, which is used to provide the source credentials
            for the STS call.
        """
        self.cache = cache
        self._load_config = load_config
        self._client_creator = client_creator
        self._profile_name = profile_name
        self._prompter = prompter
        self._loaded_config = {}
        self._credential_sourcer = credential_sourcer
        self._profile_provider_builder = profile_provider_builder
        self._visited_profiles = [self._profile_name]

    def load(self):
        self._loaded_config = self._load_config()
        profiles = self._loaded_config.get('profiles', {})
        profile = profiles.get(self._profile_name, {})
        if self._has_assume_role_config_vars(profile):
            return self._load_creds_via_assume_role(self._profile_name)

    def _has_assume_role_config_vars(self, profile):
        return self.ROLE_CONFIG_VAR in profile and self.WEB_IDENTITY_TOKE_FILE_VAR not in profile

    def _load_creds_via_assume_role(self, profile_name):
        role_config = self._get_role_config(profile_name)
        source_credentials = self._resolve_source_credentials(role_config, profile_name)
        extra_args = {}
        role_session_name = role_config.get('role_session_name')
        if role_session_name is not None:
            extra_args['RoleSessionName'] = role_session_name
        external_id = role_config.get('external_id')
        if external_id is not None:
            extra_args['ExternalId'] = external_id
        mfa_serial = role_config.get('mfa_serial')
        if mfa_serial is not None:
            extra_args['SerialNumber'] = mfa_serial
        duration_seconds = role_config.get('duration_seconds')
        if duration_seconds is not None:
            extra_args['DurationSeconds'] = duration_seconds
        fetcher = AssumeRoleCredentialFetcher(client_creator=self._client_creator, source_credentials=source_credentials, role_arn=role_config['role_arn'], extra_args=extra_args, mfa_prompter=self._prompter, cache=self.cache)
        refresher = fetcher.fetch_credentials
        if mfa_serial is not None:
            refresher = create_mfa_serial_refresher(refresher)
        return DeferredRefreshableCredentials(method=self.METHOD, refresh_using=refresher, time_fetcher=_local_now)

    def _get_role_config(self, profile_name):
        """Retrieves and validates the role configuration for the profile."""
        profiles = self._loaded_config.get('profiles', {})
        profile = profiles[profile_name]
        source_profile = profile.get('source_profile')
        role_arn = profile['role_arn']
        credential_source = profile.get('credential_source')
        mfa_serial = profile.get('mfa_serial')
        external_id = profile.get('external_id')
        role_session_name = profile.get('role_session_name')
        duration_seconds = profile.get('duration_seconds')
        role_config = {'role_arn': role_arn, 'external_id': external_id, 'mfa_serial': mfa_serial, 'role_session_name': role_session_name, 'source_profile': source_profile, 'credential_source': credential_source}
        if duration_seconds is not None:
            try:
                role_config['duration_seconds'] = int(duration_seconds)
            except ValueError:
                pass
        if credential_source is not None and source_profile is not None:
            raise InvalidConfigError(error_msg='The profile "%s" contains both source_profile and credential_source.' % profile_name)
        elif credential_source is None and source_profile is None:
            raise PartialCredentialsError(provider=self.METHOD, cred_var='source_profile or credential_source')
        elif credential_source is not None:
            self._validate_credential_source(profile_name, credential_source)
        else:
            self._validate_source_profile(profile_name, source_profile)
        return role_config

    def _validate_credential_source(self, parent_profile, credential_source):
        if self._credential_sourcer is None:
            raise InvalidConfigError(error_msg=f'The credential_source "{credential_source}" is specified in profile "{parent_profile}", but no source provider was configured.')
        if not self._credential_sourcer.is_supported(credential_source):
            raise InvalidConfigError(error_msg=f'The credential source "{credential_source}" referenced in profile "{parent_profile}" is not valid.')

    def _source_profile_has_credentials(self, profile):
        return any([self._has_static_credentials(profile), self._has_assume_role_config_vars(profile)])

    def _validate_source_profile(self, parent_profile_name, source_profile_name):
        profiles = self._loaded_config.get('profiles', {})
        if source_profile_name not in profiles:
            raise InvalidConfigError(error_msg=f'The source_profile "{source_profile_name}" referenced in the profile "{parent_profile_name}" does not exist.')
        source_profile = profiles[source_profile_name]
        if source_profile_name not in self._visited_profiles:
            return
        if source_profile_name != parent_profile_name:
            raise InfiniteLoopConfigError(source_profile=source_profile_name, visited_profiles=self._visited_profiles)
        if not self._has_static_credentials(source_profile):
            raise InfiniteLoopConfigError(source_profile=source_profile_name, visited_profiles=self._visited_profiles)

    def _has_static_credentials(self, profile):
        static_keys = ['aws_secret_access_key', 'aws_access_key_id']
        return any((static_key in profile for static_key in static_keys))

    def _resolve_source_credentials(self, role_config, profile_name):
        credential_source = role_config.get('credential_source')
        if credential_source is not None:
            return self._resolve_credentials_from_source(credential_source, profile_name)
        source_profile = role_config['source_profile']
        self._visited_profiles.append(source_profile)
        return self._resolve_credentials_from_profile(source_profile)

    def _resolve_credentials_from_profile(self, profile_name):
        profiles = self._loaded_config.get('profiles', {})
        profile = profiles[profile_name]
        if self._has_static_credentials(profile) and (not self._profile_provider_builder):
            return self._resolve_static_credentials_from_profile(profile)
        elif self._has_static_credentials(profile) or not self._has_assume_role_config_vars(profile):
            profile_providers = self._profile_provider_builder.providers(profile_name=profile_name, disable_env_vars=True)
            profile_chain = CredentialResolver(profile_providers)
            credentials = profile_chain.load_credentials()
            if credentials is None:
                error_message = 'The source profile "%s" must have credentials.'
                raise InvalidConfigError(error_msg=error_message % profile_name)
            return credentials
        return self._load_creds_via_assume_role(profile_name)

    def _resolve_static_credentials_from_profile(self, profile):
        try:
            return Credentials(access_key=profile['aws_access_key_id'], secret_key=profile['aws_secret_access_key'], token=profile.get('aws_session_token'))
        except KeyError as e:
            raise PartialCredentialsError(provider=self.METHOD, cred_var=str(e))

    def _resolve_credentials_from_source(self, credential_source, profile_name):
        credentials = self._credential_sourcer.source_credentials(credential_source)
        if credentials is None:
            raise CredentialRetrievalError(provider=credential_source, error_msg='No credentials found in credential_source referenced in profile %s' % profile_name)
        return credentials