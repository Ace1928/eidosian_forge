import re
def _formatparam(param, value=None, quote=1):
    """Convenience function to format and return a key=value pair.

    This will quote the value if needed or if quote is true.
    """
    if value is not None and len(value) > 0:
        if quote or tspecials.search(value):
            value = value.replace('\\', '\\\\').replace('"', '\\"')
            return '%s="%s"' % (param, value)
        else:
            return '%s=%s' % (param, value)
    else:
        return param