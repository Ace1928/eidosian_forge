import fixtures
class MockPatchMultiple(_Base, metaclass=_MockPatchMultipleMeta):
    """Deal with code around mock.patch.multiple."""

    def __init__(self, obj, **kwargs):
        """Initialize the mocks

        Pass name=value to replace obj.name with value.

        Pass name=Multiple.DEFAULT to replace obj.name with a
        MagicMock instance.

        :param obj: Object or name containing values being mocked.
        :type obj: str or object
        :param kwargs: names and values of attributes of obj to be mocked.

        """
        super(MockPatchMultiple, self).__init__()
        self._get_p = lambda: mock.patch.multiple(obj, **kwargs)