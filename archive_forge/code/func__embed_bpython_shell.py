from functools import wraps
def _embed_bpython_shell(namespace={}, banner=''):
    """Start a bpython shell"""
    import bpython

    @wraps(_embed_bpython_shell)
    def wrapper(namespace=namespace, banner=''):
        bpython.embed(locals_=namespace, banner=banner)
    return wrapper