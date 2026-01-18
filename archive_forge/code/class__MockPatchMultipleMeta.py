import fixtures
class _MockPatchMultipleMeta(type):
    """Arrange for lazy loading of MockPatchMultiple.DEFAULT."""

    def __new__(cls, name, bases, namespace, **kwargs):
        namespace['DEFAULT'] = cls.DEFAULT
        return super().__new__(cls, name, bases, namespace, **kwargs)

    @property
    def DEFAULT(self):
        return mock.DEFAULT