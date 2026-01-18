def _init_from_text(self, text):
    parts = text.split(' ')
    for part in parts:
        key, val = part.split('=')
        setattr(self, key.lower(), val)