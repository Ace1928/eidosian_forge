import io
import tempfile
import textwrap
from oslotest import base
from oslo_config import cfg
def _write_opt_to_tmp_file(self, group, option, value):
    filename = tempfile.mktemp()
    with io.open(filename, 'w', encoding='utf-8') as f:
        f.write(textwrap.dedent('\n            [{group}]\n            {option} = {value}\n            ').format(group=group, option=option, value=value))
    return filename