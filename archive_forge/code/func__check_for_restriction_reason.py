import copy
from heat_integrationtests.functional import functional_base
def _check_for_restriction_reason(self, events, reason, num_expected=1):
    matched = [e for e in events if e.resource_status_reason == reason]
    return len(matched) == num_expected