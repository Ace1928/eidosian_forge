from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import os
import re
import textwrap
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import yaml_printer
from googlecloudsdk.core.util import files
import six
def _WriteSchema(self, message_name, spec):
    """Writes the schema in spec to the _GetSchemaFilePath() file."""
    tmp = io.StringIO()
    tmp.write('$schema: "http://json-schema.org/draft-06/schema#"\n\n')
    yaml_printer.YamlPrinter(name='yaml', projector=resource_projector.IdentityProjector(), out=tmp).Print(spec)
    content = re.sub('\n *{}\n'.format(_YAML_WORKAROUND), '\n', tmp.getvalue())
    file_path = self._GetSchemaFilePath(message_name)
    log.info('Generating JSON schema [{}].'.format(file_path))
    with files.FileWriter(file_path) as w:
        w.write(content)