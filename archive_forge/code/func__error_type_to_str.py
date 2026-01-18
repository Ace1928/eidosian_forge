import re
def _error_type_to_str(mod, type_):
    """
    This method is registered as ofp_error_type_to_str(type_) method
    into os_ken.ofproto.ofproto_v1_* modules.
    And this method returns the error type as a string value for given
    'type' defined in ofp_error_msg structure.

    Example::

        >>> ofproto.ofp_error_type_to_str(4)
        'OFPET_BAD_MATCH(4)'
    """
    return '%s(%d)' % (_get_value_name(mod, type_, 'OFPET_'), type_)