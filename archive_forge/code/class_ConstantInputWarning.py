class ConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are exactly equal."""

    def __init__(self, msg=None):
        if msg is None:
            msg = 'All values in data are exactly equal; results may not be reliable.'
        self.args = (msg,)