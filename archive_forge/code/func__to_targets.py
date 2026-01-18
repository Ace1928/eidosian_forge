from libcloud.utils.py3 import ET
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.backup.base import BackupDriver, BackupTarget, BackupTargetJob
from libcloud.backup.types import Provider, BackupTargetType
from libcloud.common.dimensiondata import (
def _to_targets(self, object):
    node_elements = object.findall(fixxpath('server', TYPES_URN))
    return [self._to_target(el) for el in node_elements]