class FakeDriver3:

    def __init__(self):
        raise ImportError('ImportError occurs in __init__')