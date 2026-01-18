import cupy
class MetricInfo:

    def __init__(self, canonical_name=None, aka=None, validator=None, types=None):
        self.canonical_name_ = canonical_name
        self.aka_ = aka
        self.validator_ = validator
        self.types_ = types