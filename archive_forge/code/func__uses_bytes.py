def _uses_bytes(self):
    if len(self.a) > 0:
        return isinstance(self.a[0], bytes)
    elif len(self.base) > 0:
        return isinstance(self.base[0], bytes)
    elif len(self.b) > 0:
        return isinstance(self.b[0], bytes)
    else:
        return False