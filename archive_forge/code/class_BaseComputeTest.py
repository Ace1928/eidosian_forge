from openstack.tests.functional import base
class BaseComputeTest(base.BaseFunctionalTest):
    _wait_for_timeout_key = 'OPENSTACKSDK_FUNC_TEST_TIMEOUT_COMPUTE'