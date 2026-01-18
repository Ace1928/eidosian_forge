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
def _output_machine_readable(groups, output_file, conf):
    """Write a machine readable sample config file

    Take the data returned by _generate_machine_readable_data and write it in
    the format specified by the format_ attribute of conf.

    :param groups: A dict of groups as returned by _get_groups.
    :param output_file: A file-like object to which the data should be written.
    :param conf: The ConfigOpts object containing the options for the
                 generator tool
    """
    output_data = _generate_machine_readable_data(groups, conf)
    if conf.format_ == 'yaml':
        yaml.SafeDumper.add_representer(_message.Message, i18n_representer)
        output_file.write(yaml.safe_dump(output_data, default_flow_style=False))
    else:
        output_file.write(json.dumps(output_data, sort_keys=True))
    output_file.write('\n')