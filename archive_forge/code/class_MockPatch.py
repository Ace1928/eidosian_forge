import fixtures
class MockPatch(_Base):
    """Deal with code around mock.patch."""

    def __init__(self, obj, new=None, **kwargs):
        super(MockPatch, self).__init__()
        if new is None:
            new = mock.DEFAULT
        self._get_p = lambda: mock.patch(obj, new, **kwargs)