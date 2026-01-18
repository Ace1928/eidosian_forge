import collections
import copy
import itertools
import json
import os
import typing
from absl import flags
from absl.testing import parameterized
import numpy as np
from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python import numpy_util
from tensorflow.dtensor.python.config import is_gpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import is_tpu_present  # pylint: disable=unused-import
from tensorflow.dtensor.python.config import preferred_device_type  # pylint: disable=unused-import
from tensorflow.dtensor.python.tests import test_backend_util
from tensorflow.dtensor.python.tests.test_backend_name import DTENSOR_TEST_UTIL_BACKEND
from tensorflow.dtensor.python.tests.test_backend_name import DTensorTestUtilBackend
from tensorflow.dtensor.python.tests.test_backend_util import DTensorTestBackendConfigurator
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test as tf_test
class DTensorBaseTest(tf_test.TestCase, parameterized.TestCase):
    """Provides comparison helper for dtensor vs local results."""

    @classmethod
    def setUpClass(cls):
        super(DTensorBaseTest, cls).setUpClass()

    def setUp(self):
        super().setUp()
        self._backend_configurator = DTensorTestBackendConfigurator(self)

    def tearDown(self):
        try:
            context.async_wait()
        finally:
            reset_dtensor()
            self._backend_configurator.tearDown()
            super().tearDown()

    @staticmethod
    def configTestMesh(device_type_mesh_map: typing.Dict[typing.Text, layout_lib.Mesh]) -> layout_lib.Mesh:
        """Configs corresponding mesh given test context.

    If runs on a CPU mesh, set virtual device on CPU.
    If runs on a GPU mesh, sets virtual device on GPU with proper memory limits.
    if runs on a TPU mesh, initializes TPU system.

    Args:
      device_type_mesh_map: A dictionary containing device_type -> mesh mapping.

    Returns:
      A properly configured mesh for use in test.
    """
        reset_context()

        def get_mesh(device_type):
            mesh = device_type_mesh_map.get(device_type, None)
            if mesh is None:
                raise ValueError('Requires a %s mesh to run test on %s.' % (device_type, device_type))
            return mesh
        mesh = None
        if is_tpu_present():
            mesh = get_mesh('TPU')
            reset_context()
            accelerator_util.initialize_accelerator_system('TPU')
        elif tf_config.list_physical_devices('GPU'):
            mesh = get_mesh('GPU')
            reset_logical_devices('GPU', np.prod(mesh.shape()))
            accelerator_util.initialize_accelerator_system('GPU')
        else:
            mesh = get_mesh('CPU')
            reset_logical_devices('CPU', np.prod(mesh.shape()))
            accelerator_util.initialize_accelerator_system('CPU')
        test_backend_util.config_test_mesh(mesh)
        return mesh

    def skipForDeviceType(self, device_type: typing.List[str], reason: str, unless_device_count_equals_to=None):
        """Skip the test for the specific device_type.

    Args:
      device_type: list of device types, one of "CPU", "GPU", or "TPU".
      reason: string that describe the reason for skipping the test.
      unless_device_count_equals_to: Optional int. This parameter only works if
        device_type is "TPU". If set, the test will be skipped unless the number
        of TPUs equals to the specified count.
    """
        physical_device_types = set([d.device_type for d in tf_config.list_physical_devices()])
        for device in device_type:
            if device == 'TPU' and is_tpu_present():
                if unless_device_count_equals_to is None:
                    self.skipTest(reason)
                elif len(list_local_logical_devices(device)) != unless_device_count_equals_to:
                    self.skipTest(reason)
            if device == 'CPU' and len(physical_device_types) == 1 and ('CPU' in physical_device_types):
                self.skipTest(reason)
            if device == 'GPU' and 'GPU' in physical_device_types:
                self.skipTest(reason)

    def skipForTfrt(self, reason: str):
        if is_tfrt_enabled():
            self.skipTest(reason)

    def skipTest(self, reason):
        if hasattr(self, '_backend_configurator'):
            self._backend_configurator.tearDown()
        super().skipTest(reason)

    def skipForPathways(self, reason: str):
        if config.backend_is_pw():
            self.skipTest(reason)

    def assertDTensorEqual(self, expected_result, expected_layout, result_dtensor, tol=DEFAULT_TOL):
        """Asserts DTensor is of the particular value."""
        if issubclass(type(result_dtensor), resource_variable_ops.BaseResourceVariable):
            result_dtensor = result_dtensor.value()
        if expected_layout is not None:
            expected_str = expected_layout.to_string()
            got_str = api.fetch_layout(result_dtensor).to_string()
            index_for_mesh = expected_str.find('mesh:')
            if index_for_mesh != -1 and got_str.find(expected_str[index_for_mesh:]) != -1:
                expected_str = expected_str[:index_for_mesh]
                got_str = got_str[:got_str.find('mesh:')]
            self.assertEqual(api.fetch_layout(result_dtensor), expected_layout, msg='=======\nexpected layout is\n  {}\n\nwhile got layout is\n  {}\n'.format(expected_str, got_str))
        layout = api.fetch_layout(result_dtensor)
        unpacked = [t.numpy() for t in api.unpack(result_dtensor)]
        self.assertAllEqual(expected_result.shape, result_dtensor.shape)
        result_dtensor = numpy_util.to_numpy(result_dtensor)
        self.assertEqual(expected_result.dtype, result_dtensor.dtype, result_dtensor)
        self.assertAllClose(expected_result, result_dtensor, atol=tol, rtol=tol)

        def hash_key(loc):
            """Hash key for Python dict."""
            d = collections.OrderedDict(sorted(loc.items(), key=lambda x: x[0]))
            return json.dumps(d)
        offset_to_mesh_loc_dict = layout.mesh.unravel_index()
        mesh_loc_to_offset_dict = {}
        for offset, loc in offset_to_mesh_loc_dict.items():
            mesh_loc_to_offset_dict[hash_key(loc)] = offset
        replicated_dims = [x for x in layout.mesh.dim_names if x not in layout.sharding_specs]
        for offset, tensor in enumerate(unpacked):
            mesh_loc = offset_to_mesh_loc_dict[offset]
            for dim_sharding in replicated_dims:
                if mesh_loc[dim_sharding] != 0:
                    mesh_loc = copy.deepcopy(mesh_loc)
                    mesh_loc[dim_sharding] = 0
                    offset = mesh_loc_to_offset_dict[hash_key(mesh_loc)]
                    self.assertAllClose(tensor, unpacked[offset])