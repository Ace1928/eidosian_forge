class NearConstantInputWarning(DegenerateDataWarning):
    """Warns when all values in data are nearly equal."""

    def __init__(self, msg=None):
        if msg is None:
            msg = 'All values in data are nearly equal; results may not be reliable.'
        self.args = (msg,)