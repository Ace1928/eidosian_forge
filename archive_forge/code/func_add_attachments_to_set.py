import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.support import exceptions
def add_attachments_to_set(self, attachments, attachment_set_id=None):
    """
        Adds one or more attachments to an attachment set. If an
        `AttachmentSetId` is not specified, a new attachment set is
        created, and the ID of the set is returned in the response. If
        an `AttachmentSetId` is specified, the attachments are added
        to the specified set, if it exists.

        An attachment set is a temporary container for attachments
        that are to be added to a case or case communication. The set
        is available for one hour after it is created; the
        `ExpiryTime` returned in the response indicates when the set
        expires. The maximum number of attachments in a set is 3, and
        the maximum size of any attachment in the set is 5 MB.

        :type attachment_set_id: string
        :param attachment_set_id: The ID of the attachment set. If an
            `AttachmentSetId` is not specified, a new attachment set is
            created, and the ID of the set is returned in the response. If an
            `AttachmentSetId` is specified, the attachments are added to the
            specified set, if it exists.

        :type attachments: list
        :param attachments: One or more attachments to add to the set. The
            limit is 3 attachments per set, and the size limit is 5 MB per
            attachment.

        """
    params = {'attachments': attachments}
    if attachment_set_id is not None:
        params['attachmentSetId'] = attachment_set_id
    return self.make_request(action='AddAttachmentsToSet', body=json.dumps(params))