from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def _set_verify_host_remotefx_capability_mocks(self, isGpuCapable=True, isSlatCapable=True):
    s3d_video_pool = self._hostutils._conn.Msvm_Synth3dVideoPool()[0]
    s3d_video_pool.IsGpuCapable = isGpuCapable
    s3d_video_pool.IsSlatCapable = isSlatCapable