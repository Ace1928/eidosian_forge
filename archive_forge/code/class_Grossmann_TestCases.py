import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.gdp import GDP_Error
from pyomo.gdp.plugins.cuttingplane import create_cuts_fme
import pyomo.opt
import pyomo.gdp.tests.models as models
from pyomo.repn import generate_standard_repn
from pyomo.gdp.tests.common_tests import diff_apply_to_and_create_using
class Grossmann_TestCases(unittest.TestCase):

    def check_cuts_valid_at_extreme_pts(self, m):
        extreme_points = [(1, 0, 2, 10), (1, 0, 0, 10), (1, 0, 0, 7), (1, 0, 2, 7), (0, 1, 8, 0), (0, 1, 8, 3), (0, 1, 10, 0), (0, 1, 10, 3)]
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            for pt in extreme_points:
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                m.disjunct1.binary_indicator_var.fix(pt[0])
                m.disjunct2.binary_indicator_var.fix(pt[1])
                check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_valid_at_extreme_pts_fme(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_cuts_valid_at_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_valid_at_extreme_pts_projection(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cuts_valid_at_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_valid_at_extreme_pts_inf_norm(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'))
        self.check_cuts_valid_at_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_fme(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 2)
        facet2_extreme_points = [(1, 0, 2, 10), (1, 0, 2, 7), (0, 1, 10, 0), (0, 1, 10, 3)]
        facet_extreme_points = [(1, 0, 2, 10), (1, 0, 0, 10), (0, 1, 8, 3), (0, 1, 10, 3)]
        for pt in facet_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertEqual(value(cuts[0].lower), value(cuts[0].body))
        for pt in facet2_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertEqual(value(cuts[1].lower), value(cuts[1].body))

    def check_cut_is_correct_facet(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 2)
        cut1_tight_points = [(1, 0, 2, 10), (0, 1, 10, 3)]
        cut2_tight_points = [(1, 0, 2, 10), (1, 0, 0, 10)]
        for pt in cut1_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body), places=6)
        for pt in cut2_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[1].lower), value(cuts[1].body), places=6)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_projection(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cut_is_correct_facet(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_inf_norm(self):
        m = models.grossmann_oneDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'), cut_filtering_threshold=0.2)
        self.check_cut_is_correct_facet(m)

    def check_cuts_valid_at_extreme_pts_rescaled(self, m):
        extreme_points = [(1, 0, 2, 127), (1, 0, 0, 127), (1, 0, 0, 117), (1, 0, 2, 117), (0, 1, 118, 0), (0, 1, 118, 3), (0, 1, 120, 0), (0, 1, 120, 3)]
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            for pt in extreme_points:
                m.x.fix(pt[2])
                m.y.fix(pt[3])
                m.disjunct1.binary_indicator_var.fix(pt[0])
                m.disjunct2.binary_indicator_var.fix(pt[1])
                check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_at_extreme_pts_rescaled_fme(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_at_extreme_pts_rescaled_projection(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cuts_valid_at_extreme_pts_rescaled_inf_norm(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'), back_off_problem_tolerance=1e-07, verbose=True)
        self.check_cuts_valid_at_extreme_pts_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_rescaled_fme(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 1)
        cut_extreme_points = [(1, 0, 2, 127), (0, 1, 120, 3)]
        for pt in cut_extreme_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body))
            self.assertLessEqual(value(cuts[0].lower), value(cuts[0].body))

    def check_cut_is_correct_facet_rescaled(self, m):
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        self.assertEqual(len(cuts), 1)
        cut_tight_points = [(1, 0, 2, 127), (0, 1, 120, 3)]
        for pt in cut_tight_points:
            m.x.fix(pt[2])
            m.y.fix(pt[3])
            m.disjunct1.binary_indicator_var.fix(pt[0])
            m.disjunct2.binary_indicator_var.fix(pt[1])
            self.assertAlmostEqual(value(cuts[0].lower), value(cuts[0].body), places=5)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_rescaled_projection(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_cut_is_correct_facet_rescaled(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_cut_is_correct_facet_rescaled_inf_norm(self):
        m = models.to_break_constraint_tolerances()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'), cut_filtering_threshold=0.1)
        self.check_cut_is_correct_facet_rescaled(m)

    def check_2disj_cuts_valid_for_extreme_pts(self, m):
        extreme_points = [(1, 0, 1, 0, 1, 7), (1, 0, 1, 0, 1, 8), (1, 0, 1, 0, 2, 7), (1, 0, 1, 0, 2, 8), (0, 1, 0, 1, 9, 2), (0, 1, 0, 1, 9, 3), (0, 1, 0, 1, 10, 2), (0, 1, 0, 1, 10, 3)]
        cuts = m._pyomo_gdp_cuttingplane_transformation.cuts
        for cut in cuts.values():
            cut_expr = cut.body
            lower = cut.lower
            upper = cut.upper
            for pt in extreme_points:
                m.x.fix(pt[4])
                m.y.fix(pt[5])
                m.disjunct1.binary_indicator_var.fix(pt[0])
                m.disjunct2.binary_indicator_var.fix(pt[1])
                m.disjunct3.binary_indicator_var.fix(pt[2])
                m.disjunct4.binary_indicator_var.fix(pt[3])
                check_validity(self, cut_expr, lower, upper)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_2disj_cuts_valid_for_extreme_pts_fme(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, create_cuts=create_cuts_fme, post_process_cut=None)
        self.check_2disj_cuts_valid_for_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_2disj_cuts_valid_for_extreme_pts_projection(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m)
        self.check_2disj_cuts_valid_for_extreme_pts(m)

    @unittest.skipIf('ipopt' not in solvers, 'Ipopt solver not available')
    def test_2disj_cuts_valid_for_extreme_pts_inf_norm(self):
        m = models.grossmann_twoDisj()
        TransformationFactory('gdp.cuttingplane').apply_to(m, norm=float('inf'))
        self.check_2disj_cuts_valid_for_extreme_pts(m)