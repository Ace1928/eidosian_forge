from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataflow import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.dataflow import dataflow_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def _validate_yaml(yaml_pipeline):
    try:
        _ = yaml.load(yaml_pipeline)
    except Exception as exn:
        raise ValueError('yaml_pipeline must be a valid yaml.') from exn