from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import re
from googlecloudsdk.calliope.exceptions import core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
from mako import runtime
from mako import template
def ProcessOutputParameters(output_file=None, output_dir=None):
    """Helper function for generating output file and directory."""
    output_file = output_file.strip() if output_file else None
    output_dir = os.path.abspath(output_dir.strip()) if output_dir else None
    dest_file = None
    dest_dir = None
    if output_file:
        if os.path.isfile(output_file):
            overwrite_prompt = '{} already exists.'.format(output_file)
            console_io.PromptContinue(overwrite_prompt, prompt_string='Do you want to overwrite?', default=True, cancel_string='Aborted script generation.', cancel_on_no=True)
        dest_file = os.path.basename(output_file)
        dest_dir = os.path.dirname(output_file) or files.GetCWD()
        if os.path.isdir(dest_dir) and (not files.HasWriteAccessInDir(dest_dir)):
            raise TerraformGenerationError('Error writing output file: {} is not writable'.format(dest_dir))
    if output_dir:
        if os.path.isdir(output_dir) and (not files.HasWriteAccessInDir(output_dir)):
            raise ValueError('Cannot write output to directory {}. Please check permissions.'.format(output_dir))
        dest_file = None
        dest_dir = output_dir
    return (dest_file, dest_dir)