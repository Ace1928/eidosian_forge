class MockedObject:
    _name: str

    def __new__(cls, *args, **kwargs):
        if not kwargs.get('_suppress_err'):
            raise NotImplementedError(f"Object '{cls._name}' was mocked out during packaging but it is being used in '__new__'. If this error is happening during 'load_pickle', please ensure that your pickled object doesn't contain any mocked objects.")
        return super().__new__(cls)

    def __init__(self, name: str, _suppress_err: bool):
        self.__dict__['_name'] = name

    def __repr__(self):
        return f'MockedObject({self._name})'