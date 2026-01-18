import sys
from os_win._i18n import _
class SCSIIdDescriptorParsingError(Invalid):
    msg_fmt = _('Parsing SCSI identification descriptor failed. Reason: %(reason)s.')