import sys
from os_win._i18n import _
class SCSIPageParsingError(Invalid):
    msg_fmt = _('Parsing SCSI Page %(page)s failed. Reason: %(reason)s.')