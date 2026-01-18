class ApplicationError(Error):
    """Raised by APIProxy in the event of an application-level error."""

    def __init__(self, application_error, error_detail=''):
        self.application_error = application_error
        self.error_detail = error_detail
        Error.__init__(self, application_error)

    def __str__(self):
        return 'ApplicationError: %d %s' % (self.application_error, self.error_detail)