class InstanceStatusSet(list):
    """
    A list object that contains the results of a call to
    DescribeInstanceStatus request.  Each element of the
    list will be an InstanceStatus object.

    :ivar next_token: If the response was truncated by
        the EC2 service, the next_token attribute of the
        object will contain the string that needs to be
        passed in to the next request to retrieve the next
        set of results.
    """

    def __init__(self, connection=None):
        list.__init__(self)
        self.connection = connection
        self.next_token = None

    def startElement(self, name, attrs, connection):
        if name == 'item':
            status = InstanceStatus()
            self.append(status)
            return status
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'nextToken':
            self.next_token = value
        setattr(self, name, value)