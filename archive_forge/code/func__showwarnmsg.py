import sys
def _showwarnmsg(msg):
    """Hook to write a warning to a file; replace if you like."""
    try:
        sw = showwarning
    except NameError:
        pass
    else:
        if sw is not _showwarning_orig:
            if not callable(sw):
                raise TypeError('warnings.showwarning() must be set to a function or method')
            sw(msg.message, msg.category, msg.filename, msg.lineno, msg.file, msg.line)
            return
    _showwarnmsg_impl(msg)