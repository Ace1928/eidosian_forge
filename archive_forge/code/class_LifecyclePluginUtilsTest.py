from unittest import mock
from heat.common import lifecycle_plugin_utils
from heat.engine import lifecycle_plugin
from heat.engine import resources
from heat.tests import common
class LifecyclePluginUtilsTest(common.HeatTestCase):
    """Basic tests for :module:'heat.common.lifecycle_plugin_utils'.

    Basic tests for the helper methods in
    :module:'heat.common.lifecycle_plugin_utils'.
    """

    def tearDown(self):
        super(LifecyclePluginUtilsTest, self).tearDown()
        lifecycle_plugin_utils.pp_class_instances = None

    def mock_lcp_class_map(self, lcp_mappings):
        self.mock_get_plugins = self.patchobject(resources.global_env(), 'get_stack_lifecycle_plugins', return_value=lcp_mappings)
        lifecycle_plugin_utils.pp_class_instances = None

    def test_get_plug_point_class_instances(self):
        """Tests the get_plug_point_class_instances function."""
        lcp_mappings = [('A::B::C1', TestLifecycleCallout1)]
        self.mock_lcp_class_map(lcp_mappings)
        pp_cinstances = lifecycle_plugin_utils.get_plug_point_class_instances()
        self.assertIsNotNone(pp_cinstances)
        self.assertTrue(self.is_iterable(pp_cinstances), 'not iterable: %s' % pp_cinstances)
        self.assertEqual(1, len(pp_cinstances))
        self.assertEqual(TestLifecycleCallout1, pp_cinstances[0].__class__)
        self.mock_get_plugins.assert_called_once_with()

    def test_do_pre_and_post_callouts(self):
        lcp_mappings = [('A::B::C1', TestLifecycleCallout1)]
        self.mock_lcp_class_map(lcp_mappings)
        mc = mock.Mock()
        mc.__setattr__('pre_counter_for_unit_test', 0)
        mc.__setattr__('post_counter_for_unit_test', 0)
        ms = mock.Mock()
        ms.__setattr__('action', 'A')
        lifecycle_plugin_utils.do_pre_ops(mc, ms, None, None)
        self.assertEqual(1, mc.pre_counter_for_unit_test)
        lifecycle_plugin_utils.do_post_ops(mc, ms, None, None)
        self.assertEqual(1, mc.post_counter_for_unit_test)
        self.mock_get_plugins.assert_called_once_with()

    def test_class_instantiation_and_sorting(self):
        lcp_mappings = []
        self.mock_lcp_class_map(lcp_mappings)
        pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
        self.assertEqual(0, len(pp_cis))
        self.mock_get_plugins.assert_called_once_with()
        lcp_mappings = [('A::B::C2', TestLifecycleCallout2), ('A::B::C1', TestLifecycleCallout1)]
        self.mock_lcp_class_map(lcp_mappings)
        pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
        self.assertEqual(2, len(pp_cis))
        self.assertEqual(100, pp_cis[0].get_ordinal())
        self.assertEqual(101, pp_cis[1].get_ordinal())
        self.assertEqual(TestLifecycleCallout1, pp_cis[0].__class__)
        self.assertEqual(TestLifecycleCallout2, pp_cis[1].__class__)
        self.mock_get_plugins.assert_called_once_with()
        lcp_mappings = [('A::B::C1', TestLifecycleCallout1), ('A::B::C2', TestLifecycleCallout2)]
        self.mock_lcp_class_map(lcp_mappings)
        pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
        self.assertEqual(2, len(pp_cis))
        self.assertEqual(100, pp_cis[0].get_ordinal())
        self.assertEqual(101, pp_cis[1].get_ordinal())
        self.assertEqual(TestLifecycleCallout1, pp_cis[0].__class__)
        self.assertEqual(TestLifecycleCallout2, pp_cis[1].__class__)
        self.mock_get_plugins.assert_called_once_with()
        lcp_mappings = [('A::B::C2', TestLifecycleCallout2), ('A::B::C3', TestLifecycleCallout3), ('A::B::C1', TestLifecycleCallout1)]
        self.mock_lcp_class_map(lcp_mappings)
        pp_cis = lifecycle_plugin_utils.get_plug_point_class_instances()
        self.assertEqual(3, len(pp_cis))
        self.assertEqual(100, pp_cis[2].get_ordinal())
        self.assertEqual(101, pp_cis[0].get_ordinal())
        self.assertEqual(TestLifecycleCallout2, pp_cis[0].__class__)
        self.assertEqual(TestLifecycleCallout3, pp_cis[1].__class__)
        self.assertEqual(TestLifecycleCallout1, pp_cis[2].__class__)
        self.mock_get_plugins.assert_called_once_with()

    def test_do_pre_op_failure(self):
        lcp_mappings = [('A::B::C5', TestLifecycleCallout1), ('A::B::C4', TestLifecycleCallout4)]
        self.mock_lcp_class_map(lcp_mappings)
        mc = mock.Mock()
        mc.__setattr__('pre_counter_for_unit_test', 0)
        mc.__setattr__('post_counter_for_unit_test', 0)
        ms = mock.Mock()
        ms.__setattr__('action', 'A')
        failed = False
        try:
            lifecycle_plugin_utils.do_pre_ops(mc, ms, None, None)
        except Exception:
            failed = True
        self.assertTrue(failed)
        self.assertEqual(1, mc.pre_counter_for_unit_test)
        self.assertEqual(1, mc.post_counter_for_unit_test)
        self.mock_get_plugins.assert_called_once_with()

    def test_do_post_op_failure(self):
        lcp_mappings = [('A::B::C1', TestLifecycleCallout1), ('A::B::C5', TestLifecycleCallout5)]
        self.mock_lcp_class_map(lcp_mappings)
        mc = mock.Mock()
        mc.__setattr__('pre_counter_for_unit_test', 0)
        mc.__setattr__('post_counter_for_unit_test', 0)
        ms = mock.Mock()
        ms.__setattr__('action', 'A')
        lifecycle_plugin_utils.do_post_ops(mc, ms, None, None)
        self.assertEqual(1, mc.post_counter_for_unit_test)
        self.mock_get_plugins.assert_called_once_with()

    def test_exercise_base_lifecycle_plugin_class(self):
        lcp = lifecycle_plugin.LifecyclePlugin()
        ordinal = lcp.get_ordinal()
        lcp.do_pre_op(None, None, None)
        lcp.do_post_op(None, None, None)
        self.assertEqual(100, ordinal)

    def is_iterable(self, obj):
        if not object:
            return False
        if isinstance(obj, str):
            return False
        try:
            for m in obj:
                break
        except TypeError:
            return False
        return True