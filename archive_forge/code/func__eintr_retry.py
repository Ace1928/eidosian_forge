def _eintr_retry(*args):
    """restart a system call interrupted by EINTR"""
    while True:
        try:
            return real_select(*args)
        except (OSError, select.error) as ex:
            if ex.args[0] != errno.EINTR:
                raise