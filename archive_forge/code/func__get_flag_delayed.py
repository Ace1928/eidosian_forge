def _get_flag_delayed(self, current_flag):
    """Implement the "delayed" advancement of the global stale flag value

        This will continue to return the current value of the state flag
        until the first non-stale variable is updated (that it, it is
        passed the current stale flag when called).  This allows for
        updating stale variable values without incrementing the global
        stale flag, but will mark everything as stale as soon as a
        non-stale variable value is changed.

        """
    if current_flag == self._current:
        self._current += 1
        setattr(self, 'get_flag', getattr(self, '_get_flag'))
    return self._current