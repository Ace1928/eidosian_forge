import pickle
import pytest
import numpy as np
@pytest.mark.parametrize('args', [(2, 1, None), (2, 1, 'test_prefix'), ('test message',)])
class TestAxisError:

    def test_attr(self, args):
        """Validate attribute types."""
        exc = np.AxisError(*args)
        if len(args) == 1:
            assert exc.axis is None
            assert exc.ndim is None
        else:
            axis, ndim, *_ = args
            assert exc.axis == axis
            assert exc.ndim == ndim

    def test_pickling(self, args):
        """Test that `AxisError` can be pickled."""
        exc = np.AxisError(*args)
        exc2 = pickle.loads(pickle.dumps(exc))
        assert type(exc) is type(exc2)
        for name in ('axis', 'ndim', 'args'):
            attr1 = getattr(exc, name)
            attr2 = getattr(exc2, name)
            assert attr1 == attr2, name