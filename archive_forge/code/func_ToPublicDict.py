def ToPublicDict(self):
    out = {}
    for k, v in list(self.__dict__.items()):
        if not k.startswith('internal_'):
            out[k] = v
    return out