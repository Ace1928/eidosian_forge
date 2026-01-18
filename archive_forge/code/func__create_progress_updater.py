import logging
import tarfile
from eventlet import timeout
from oslo_utils import units
from oslo_vmware._i18n import _
from oslo_vmware.common import loopingcall
from oslo_vmware import constants
from oslo_vmware import exceptions
from oslo_vmware import image_util
from oslo_vmware.objects import datastore as ds_obj
from oslo_vmware import rw_handles
from oslo_vmware import vim_util
def _create_progress_updater(handle):
    if isinstance(handle, rw_handles.VmdkHandle):
        updater = loopingcall.FixedIntervalLoopingCall(handle.update_progress)
        updater.start(interval=NFC_LEASE_UPDATE_PERIOD)
        return updater