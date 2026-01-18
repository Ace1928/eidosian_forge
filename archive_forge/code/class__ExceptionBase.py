from os_ken import exception
class _ExceptionBase(exception.OSKenException):

    def __init__(self, result):
        self.result = result
        super(_ExceptionBase, self).__init__(result=result)