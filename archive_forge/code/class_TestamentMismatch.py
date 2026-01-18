class TestamentMismatch(BzrError):
    _fmt = 'Testament did not match expected value.\n       For revision_id {%(revision_id)s}, expected {%(expected)s}, measured\n       {%(measured)s}'

    def __init__(self, revision_id, expected, measured):
        self.revision_id = revision_id
        self.expected = expected
        self.measured = measured