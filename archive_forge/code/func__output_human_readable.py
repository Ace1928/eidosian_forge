import collections
import copy
import importlib.metadata
import json
import logging
import operator
import sys
import yaml
from oslo_config import cfg
from oslo_i18n import _message
import stevedore.named  # noqa
def _output_human_readable(namespaces, output_file):
    """Write an RST formated version of the docs for the options.

    :param groups: A list of the namespaces to use for discovery.
    :param output_file: A file-like object to which the data should be written.
    """
    try:
        from oslo_config import sphinxext
    except ImportError:
        raise RuntimeError('Could not import sphinxext. Please install Sphinx and try again.')
    output_data = list(sphinxext._format_option_help(LOG, namespaces, False))
    output_file.write('\n'.join(output_data))