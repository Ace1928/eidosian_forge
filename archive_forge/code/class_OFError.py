from os_ken import exception
class OFError(_ExceptionBase):
    """OFPErrorMsg is received."""
    message = 'OpenFlow errors %(result)s'