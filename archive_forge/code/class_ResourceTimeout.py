class ResourceTimeout(CoreException):

    def __init__(self, message='', result=None):
        self.result = result or {}
        super().__init__(message)