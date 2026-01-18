class _Run:

    def __init__(self, value, count):
        self.value = value
        self.count = count

    def __repr__(self):
        return f'Run({self.value}, {self.count})'