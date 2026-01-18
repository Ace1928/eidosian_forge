import importlib.metadata
import logging
import re
import sys
import yaml
from oslo_config import cfg
from oslo_config import generator
def _register_cli_opts(conf):
    """Register the formatter's CLI options with a ConfigOpts instance.

    Note, this must be done before the ConfigOpts instance is called to parse
    the configuration.

    :param conf: a ConfigOpts instance
    :raises: DuplicateOptError, ArgsAlreadyParsedError
    """
    conf.register_cli_opts(_validator_opts)