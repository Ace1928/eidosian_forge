import os
from cupy.testing._pytest_impl import is_available, check_available
def _dummy_callable(*args, **kwargs):
    check_available('pytest attributes')
    assert False