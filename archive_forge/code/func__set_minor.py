def _set_minor(self, s):
    s = s.lower()
    if not is_token(s):
        raise ValueError('Minor media type contains an invalid character')
    self._minor = s