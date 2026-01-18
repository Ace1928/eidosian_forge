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
def _start_transfer(read_handle, write_handle, timeout_secs):
    read_updater = _create_progress_updater(read_handle)
    write_updater = _create_progress_updater(write_handle)
    timer = timeout.Timeout(timeout_secs)
    try:
        while True:
            data = read_handle.read(CHUNK_SIZE)
            if not data:
                break
            write_handle.write(data)
    except timeout.Timeout as excep:
        msg = _('Timeout, read_handle: "%(src)s", write_handle: "%(dest)s"') % {'src': read_handle, 'dest': write_handle}
        LOG.exception(msg)
        raise exceptions.ImageTransferException(msg, excep)
    except Exception as excep:
        msg = _('Error, read_handle: "%(src)s", write_handle: "%(dest)s"') % {'src': read_handle, 'dest': write_handle}
        LOG.exception(msg)
        raise exceptions.ImageTransferException(msg, excep)
    finally:
        timer.cancel()
        if read_updater:
            read_updater.stop()
        if write_updater:
            write_updater.stop()
        read_handle.close()
        write_handle.close()