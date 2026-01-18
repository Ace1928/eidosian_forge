class ScalarValue(Value):
    combinable = True

    def __init__(self, v):
        self.v = str(v)

    def to_string(self):
        return self.v