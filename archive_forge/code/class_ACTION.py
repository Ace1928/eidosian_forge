import hashlib
from base64 import b64encode
from libcloud.utils.py3 import ET, b, next, httplib
from libcloud.common.base import XmlResponse, ConnectionUserAndKey
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import (
from libcloud.compute.providers import Provider
class ACTION:
    """
    All actions, except RESUME, only apply when the VM is in the "Running"
    state.
    """
    STOP = 'STOPPED'
    '\n    The VM is stopped, and its memory state stored to a checkpoint file. VM\n    state, and disk image, are transferred back to the front-end. Resuming\n    the VM requires the VM instance to be re-scheduled.\n    '
    SUSPEND = 'SUSPENDED'
    '\n    The VM is stopped, and its memory state stored to a checkpoint file. The VM\n    state, and disk image, are left on the host to be resumed later. Resuming\n    the VM does not require the VM to be re-scheduled. Rather, after\n    suspending, the VM resources are reserved for later resuming.\n    '
    RESUME = 'RESUME'
    "\n    The VM is resumed using the saved memory state from the checkpoint file,\n    and the VM's disk image. The VM is either started immediately, or\n    re-scheduled depending on how it was suspended.\n    "
    CANCEL = 'CANCEL'
    '\n    The VM is forcibly shutdown, its memory state is deleted. If a persistent\n    disk image was used, that disk image is transferred back to the front-end.\n    Any non-persistent disk images are deleted.\n    '
    SHUTDOWN = 'SHUTDOWN'
    '\n    The VM is gracefully shutdown by sending the ACPI signal. If the VM does\n    not shutdown, then it is considered to still be running. If successfully,\n    shutdown, its memory state is deleted. If a persistent disk image was used,\n    that disk image is transferred back to the front-end. Any non-persistent\n    disk images are deleted.\n    '
    REBOOT = 'REBOOT'
    '\n    Introduced in OpenNebula v3.2.\n\n    The VM is gracefully restarted by sending the ACPI signal.\n    '
    DONE = 'DONE'
    '\n    The VM is forcibly shutdown, its memory state is deleted. If a persistent\n    disk image was used, that disk image is transferred back to the front-end.\n    Any non-persistent disk images are deleted.\n    '