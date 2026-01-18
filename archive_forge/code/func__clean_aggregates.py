from novaclient.tests.functional import base
def _clean_aggregates(self):
    for a in (self.agg1, self.agg2):
        try:
            self.nova('aggregate-delete', params=a)
        except Exception:
            pass