from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os.path
from googlecloudsdk.core import branding
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import name_parsing
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import files
from mako import runtime
from mako import template
def _RenderFile(file_path, file_template, context, enable_overwrites):
    """Renders a file to given path using the provided template and context."""
    render_file = False
    overwrite = False
    if not os.path.exists(file_path):
        render_file = True
    elif enable_overwrites:
        render_file = True
        overwrite = True
    if render_file:
        log.status.Print(' -- Generating: File: [{}], Overwrite: [{}]'.format(file_path, overwrite))
        with files.FileWriter(file_path, create_path=True) as f:
            ctx = runtime.Context(f, **context)
            file_template.render_context(ctx)
    else:
        log.status.Print(' >> Skipped: File: [{}] --'.format(file_path))