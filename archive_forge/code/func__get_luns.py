from __future__ import annotations
import copy
from typing import Any, Generator  # noqa: H301
from os_brick.initiator import initiator_connector
@staticmethod
def _get_luns(con_props: dict, iqns=None) -> list:
    luns = con_props.get('target_luns')
    num_luns = len(con_props['target_iqns']) if iqns is None else len(iqns)
    return luns or [con_props['target_lun']] * num_luns