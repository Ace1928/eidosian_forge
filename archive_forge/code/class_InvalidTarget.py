class InvalidTarget(MessagingException, ValueError):
    """Raised if a target does not meet certain pre-conditions."""

    def __init__(self, msg, target):
        msg = msg + ':' + str(target)
        super(InvalidTarget, self).__init__(msg)
        self.target = target