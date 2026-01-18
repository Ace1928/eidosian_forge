class ResultEntry(dict):
    """
    The result (successful or unsuccessful) of a single
    message within a send_message_batch request.

    In the case of a successful result, this dict-like
    object will contain the following items:

    :ivar id: A string containing the user-supplied ID of the message.
    :ivar message_id: A string containing the SQS ID of the new message.
    :ivar message_md5: A string containing the MD5 hash of the message body.

    In the case of an error, this object will contain the following
    items:

    :ivar id: A string containing the user-supplied ID of the message.
    :ivar sender_fault: A boolean value.
    :ivar error_code: A string containing a short description of the error.
    :ivar error_message: A string containing a description of the error.
    """

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Id':
            self['id'] = value
        elif name == 'MessageId':
            self['message_id'] = value
        elif name == 'MD5OfMessageBody':
            self['message_md5'] = value
        elif name == 'SenderFault':
            self['sender_fault'] = value
        elif name == 'Code':
            self['error_code'] = value
        elif name == 'Message':
            self['error_message'] = value