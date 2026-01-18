from .. import sage_helper
def Vector3(*args, **kwargs):
    ans = Vector(*args, **kwargs)
    assert len(ans) == 3
    return ans