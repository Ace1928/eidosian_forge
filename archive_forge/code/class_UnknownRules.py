import configobj
from . import bedding, cmdline, errors, globbing, osutils
class UnknownRules(errors.BzrError):
    _fmt = 'Unknown rules detected: %(unknowns_str)s.'

    def __init__(self, unknowns):
        errors.BzrError.__init__(self, unknowns_str=', '.join(unknowns))