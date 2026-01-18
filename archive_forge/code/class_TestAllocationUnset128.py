import uuid
from osc_placement.tests.functional import base
class TestAllocationUnset128(TestAllocationUnset112):
    """Tests allocation unset command with --os-placement-api-version 1.28.

    The 1.28 microversion adds the consumer_generation parameter to the
    GET and PUT /allocations/{consumer_id} APIs.
    """
    VERSION = '1.28'