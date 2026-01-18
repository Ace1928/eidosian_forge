import re
def _error_code_to_str(mod, type_, code):
    """
    This method is registered as ofp_error_code_to_str(type_, code) method
    into os_ken.ofproto.ofproto_v1_* modules.
    And this method returns the error code as a string value for given
    'type' and 'code' defined in ofp_error_msg structure.

    Example::

        >>> ofproto.ofp_error_code_to_str(4, 9)
        'OFPBMC_BAD_PREREQ(9)'
    """
    _, c_name = _get_error_names(mod, type_, code)
    return '%s(%d)' % (c_name, code)