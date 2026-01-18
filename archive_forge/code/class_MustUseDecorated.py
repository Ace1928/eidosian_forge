class MustUseDecorated(Exception):
    _fmt = 'A decorating function has requested its original command be used.'