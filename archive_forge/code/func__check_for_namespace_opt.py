import logging
import sys
import textwrap
import warnings
import yaml
from oslo_config import cfg
from oslo_serialization import jsonutils
import stevedore
from oslo_policy import policy
def _check_for_namespace_opt(conf):
    if conf.namespace is None:
        raise cfg.RequiredOptError('namespace', 'DEFAULT')