import copy
import itertools
from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import config
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_types
def _add_standard_extra_specs_to_dict(self, extra_specs, create_from_snapshot=None, revert_to_snapshot=None, mount_snapshot=None):
    if all((spec is None for spec in [create_from_snapshot, revert_to_snapshot, mount_snapshot])):
        return extra_specs
    extra_specs = extra_specs or {}
    if create_from_snapshot is not None:
        extra_specs['create_share_from_snapshot_support'] = create_from_snapshot
    if revert_to_snapshot is not None:
        extra_specs['revert_to_snapshot_support'] = revert_to_snapshot
    if mount_snapshot is not None:
        extra_specs['mount_snapshot_support'] = mount_snapshot
    return extra_specs