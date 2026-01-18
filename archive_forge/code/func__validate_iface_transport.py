from __future__ import annotations
from collections import defaultdict
import copy
import glob
import os
import re
import time
from typing import Any, Iterable, Optional, Union  # noqa: H301
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import strutils
from os_brick import exception
from os_brick import executor
from os_brick.i18n import _
from os_brick import initiator
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator import utils as initiator_utils
from os_brick import utils
def _validate_iface_transport(self, transport_iface: str) -> str:
    """Check that given iscsi_iface uses only supported transports

        Accepted transport names for provided iface param are
        be2iscsi, bnx2i, cxgb3i, cxgb4i, default, qla4xxx, ocs, iser or tcp.
        Note the difference between transport and iface;
        unlike default(iscsi_tcp)/iser, this is not one and the same for
        offloaded transports, where the default format is
        transport_name.hwaddress

        :param transport_iface: The iscsi transport type.
        :type transport_iface: str
        :returns: str
        """
    if transport_iface in ['default', 'iser']:
        return transport_iface
    out = self._run_iscsiadm_bare(['-m', 'iface', '-I', transport_iface], check_exit_code=[0, 2, 6])[0] or ''
    LOG.debug('iscsiadm %(iface)s configuration: stdout=%(out)s.', {'iface': transport_iface, 'out': out})
    for data in [line.split() for line in out.splitlines()]:
        if data[0] == 'iface.transport_name':
            if data[2] in self.supported_transports:
                return transport_iface
    LOG.warning('No useable transport found for iscsi iface %s. Falling back to default transport.', transport_iface)
    return 'default'