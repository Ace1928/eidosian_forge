from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.run import k8s_object
@property
def active_revisions(self):
    """Return the revisions whose traffic target is positive."""
    revisions = {}
    for traffic_target in self._m.status.traffic:
        if traffic_target.percent:
            revisions[traffic_target.revisionName] = traffic_target.percent
    return revisions