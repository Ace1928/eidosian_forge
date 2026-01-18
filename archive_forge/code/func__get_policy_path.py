import collections.abc
import copy
import logging
import os
import typing as ty
import warnings
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslo_utils import strutils
import yaml
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_policy import _parser
from oslo_policy import opts
def _get_policy_path(self, path):
    """Locate the policy YAML/JSON data file/path.

        :param path: It's value can be a full path or related path. When
            full path specified, this function just returns the full path. When
            related path specified, this function will search configuration
            directories to find one that exists.

        :returns: The policy path
        :raises: ConfigFilesNotFoundError if the file/path couldn't be located.
        """
    policy_path = self.conf.find_file(path)
    if policy_path:
        return policy_path
    raise cfg.ConfigFilesNotFoundError((path,))