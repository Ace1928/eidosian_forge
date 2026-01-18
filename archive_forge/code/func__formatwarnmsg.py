import sys
def _formatwarnmsg(msg):
    """Function to format a warning the standard way."""
    try:
        fw = formatwarning
    except NameError:
        pass
    else:
        if fw is not _formatwarning_orig:
            return fw(msg.message, msg.category, msg.filename, msg.lineno, msg.line)
    return _formatwarnmsg_impl(msg)