from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base as calliope_base
from googlecloudsdk.core.util import files
def AddTerraformGenerateImportArgs(parser):
    """Arguments for resource-config terraform generate-import command."""
    input_path_help = 'Path to a Terrafrom formatted (.tf) resource file or directory of files exported via. `gcloud alpha resource-config bulk-export` or resource surface specific `config export` command.'
    input_path = calliope_base.Argument('INPUT_PATH', type=files.ExpandHomeAndVars, help=input_path_help)
    output_args = calliope_base.ArgumentGroup(category='OUTPUT DESTINATION', mutex=True, help='Specify the destination of the generated script.')
    file_spec_group = calliope_base.ArgumentGroup(help='Specify the exact filenames for the output import script and module files.')
    file_spec_group.AddArgument(calliope_base.Argument('--output-script-file', required=False, type=files.ExpandHomeAndVars, help='Specify the full path path for generated import script. If not set, a default filename of the form `terraform_import_YYYYMMDD-HH-MM-SS.sh|cmd` will be generated.'))
    file_spec_group.AddArgument(calliope_base.Argument('--output-module-file', required=False, type=files.ExpandHomeAndVars, help='Specify the full path path for generated terraform module file. If not set, a default filename of `gcloud-export-modules.tf` will be generated.'))
    output_args.AddArgument(file_spec_group)
    output_args.AddArgument(calliope_base.Argument('--output-dir', required=False, type=files.ExpandHomeAndVars, help='Specify the output directory only for the generated import script. If specified directory does not exists it will be created. Generated script will have a default name of the form `terraform_import_YYYYMMDD-HH-MM-SS.sh|cmd`'))
    input_path.AddToParser(parser)
    output_args.AddToParser(parser)