from heat.engine.resources import signal_responder
from heat.engine.resources import wait_condition as wc_base
from heat.engine import support
class WaitConditionHandle(wc_base.BaseWaitConditionHandle):
    """AWS WaitConditionHandle resource.

    the main point of this class is to :
    have no dependencies (so the instance can reference it)
    generate a unique url (to be returned in the reference)
    then the cfn-signal will use this url to post to and
    WaitCondition will poll it to see if has been written to.
    """
    support_status = support.SupportStatus(version='2014.1')
    METADATA_KEYS = DATA, REASON, STATUS, UNIQUE_ID = ('Data', 'Reason', 'Status', 'UniqueId')

    def get_reference_id(self):
        if self.resource_id:
            wc = signal_responder.WAITCONDITION
            return str(self._get_ec2_signed_url(signal_type=wc))
        else:
            return str(self.name)

    def metadata_update(self, new_metadata=None):
        """DEPRECATED. Should use handle_signal instead."""
        self.handle_signal(details=new_metadata)

    def handle_signal(self, details=None):
        """Validate and update the resource metadata.

        metadata must use the following format::

            {
                "Status" : "Status (must be SUCCESS or FAILURE)",
                "UniqueId" : "Some ID, should be unique for Count>1",
                "Data" : "Arbitrary Data",
                "Reason" : "Reason String"
            }
        """
        if details is None:
            return
        return super(WaitConditionHandle, self).handle_signal(details)