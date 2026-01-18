from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import datetime
import fnmatch
import json
from googlecloudsdk.command_lib.anthos.config.sync.common import exceptions
from googlecloudsdk.command_lib.anthos.config.sync.common import utils
from googlecloudsdk.core import log
def _GetOciKey(obj):
    """Hash the Oci specification of the given RepoSync|RootSync object."""
    image = _GetPathValue(obj, ['spec', 'oci', 'image'])
    if not image:
        return ''
    directory = _GetPathValue(obj, ['spec', 'oci', 'dir'], '.')
    if directory in {'', '.', '/'}:
        oci_str = image.rstrip('/')
    else:
        oci_str = '{image}/{directory}'.format(image=image.rstrip('/'), directory=directory.lstrip('/'))
    return oci_str