from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import copy
import io
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
def WriteToDisk(self):
    """Overwrite Original Yaml File."""
    if not self.file_path:
        raise YamlConfigFileError('Could Not Write To Config File: Path Is Empty')
    out_file_buf = io.BytesIO()
    tmp_yaml_buf = io.TextIOWrapper(out_file_buf, newline='\n', encoding='utf-8')
    yaml.dump_all_round_trip([x.content for x in self.data], stream=tmp_yaml_buf)
    with files.BinaryFileWriter(self.file_path) as f:
        tmp_yaml_buf.seek(0)
        f.write(out_file_buf.getvalue())