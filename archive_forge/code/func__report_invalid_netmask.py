import functools
@classmethod
def _report_invalid_netmask(cls, netmask_str):
    msg = '%r is not a valid netmask' % netmask_str
    raise NetmaskValueError(msg) from None