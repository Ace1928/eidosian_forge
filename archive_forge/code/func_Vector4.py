from .. import sage_helper
def Vector4(*args, **kwargs):
    ans = Vector(*args, **kwargs)
    assert len(ans) == 4
    return ans