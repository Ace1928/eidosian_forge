import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
class NonlinearConvex_TwoCircles(unittest.TestCase):

    def check_cuts_valid_for_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1)
        m.x.fix(2)
        m.y.fix(7)
        m.upper_circle.indicator_var.fix(True)
        m.lower_circle.indicator_var.fix(False)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, verbose=True)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'), verbose=True)
        self.check_cuts_valid_for_optimal(m)

    def check_cuts_valid_on_facet_containing_optimal(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1)
        m.x.fix(5)
        m.y.fix(3)
        m.upper_circle.indicator_var.fix(False)
        m.lower_circle.indicator_var.fix(True)
        for i in range(len(cuts)):
            self.assertTrue(value(cuts[i].expr))

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, verbose=True)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_on_facet_containing_optimal_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'), verbose=True)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    def check_cuts_valid_for_other_extreme_points(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertGreaterEqual(len(cuts), 1)
        m.x.fix(3)
        m.y.fix(1)
        m.upper_circle.indicator_var.fix(True)
        m.lower_circle.indicator_var.fix(False)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)
        m.x.fix(0)
        m.y.fix(5)
        m.upper_circle.indicator_var.fix(False)
        m.lower_circle.indicator_var.fix(True)
        for i in range(len(cuts)):
            self.assertGreaterEqual(value(cuts[i].body), 0)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, create_cuts=create_cuts_fme, verbose=True)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=1000000.0, norm=float('inf'), cut_filtering_threshold=0.5)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, norm=float('inf'))
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimal_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimalFacet_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimalFacet_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_optimalFacet_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, norm=float('inf'))
        self.check_cuts_valid_on_facet_containing_optimal(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points_tighter_m(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points_tighter_m_fme(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, create_cuts=create_cuts_fme)
        self.check_cuts_valid_for_other_extreme_points(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_for_other_extreme_points_tighter_m_inf_norm(self):
        m = models.twoDisj_twoCircles_easy()
        TransformationFactory('gdp.cuttingplane').apply_to(m, bigM=83, norm=float('inf'), cut_filtering_threshold=0.5)
        self.check_cuts_valid_for_other_extreme_points(m)