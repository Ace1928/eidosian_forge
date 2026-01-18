import testtools
def _assertRequestId(self, obj):
    self.assertIsNotNone(getattr(obj, 'request_ids', None))
    self.assertEqual(['req-1234'], obj.request_ids)