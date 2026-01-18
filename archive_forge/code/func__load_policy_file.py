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
def _load_policy_file(self, path, force_reload, overwrite=True):
    """Load policy rules from the specified policy file.

        :param path: A path of the policy file to load rules from.
        :param force_reload: Forcefully reload the policy file content.
        :param overwrite: Replace policy rules instead of updating them.
        :return: A bool indicating whether rules have been changed or not.
        """
    rules_changed = False
    reloaded, data = _cache_handler.read_cached_file(self._file_cache, path, force_reload=force_reload)
    if reloaded or not self.rules:
        rules = Rules.load(data, self.default_rule)
        self.set_rules(rules, overwrite=overwrite, use_conf=True)
        rules_changed = True
        self._record_file_rules(data, overwrite)
        LOG.debug('Reloaded policy file: %(path)s', {'path': path})
    return rules_changed