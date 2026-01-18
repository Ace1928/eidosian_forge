import configparser
import getpass
import logging
import os
from typing import NamedTuple, Optional, Tuple
from mlflow.environment_variables import (
from mlflow.exceptions import MlflowException
from mlflow.utils.rest_utils import MlflowHostCreds
def _databricks_login(interactive):
    """Set up databricks authentication."""
    try:
        _validate_databricks_auth()
        return
    except Exception:
        if interactive:
            _logger.info('No valid Databricks credentials found, please enter your credentials...')
        else:
            raise MlflowException('No valid Databricks credentials found while running in non-interactive mode.')
    while True:
        host = input('Databricks Host (should begin with https://): ')
        if not host.startswith('https://'):
            _logger.error('Invalid host: {host}, host must begin with https://, please retry.')
        break
    profile = {'host': host}
    if 'community' in host:
        username = input('Username: ')
        password = getpass.getpass('Password: ')
        profile['username'] = username
        profile['password'] = password
    else:
        token = getpass.getpass('Token: ')
        profile['token'] = token
    file_name = os.environ.get('DATABRICKS_CONFIG_FILE', f'{os.path.expanduser('~')}/.databrickscfg')
    profile_name = os.environ.get('DATABRICKS_CONFIG_PROFILE', 'DEFAULT')
    _overwrite_or_create_databricks_profile(file_name, profile, profile_name)
    try:
        _validate_databricks_auth()
    except Exception as e:
        raise MlflowException(f'`mlflow.login()` failed with error: {e}')