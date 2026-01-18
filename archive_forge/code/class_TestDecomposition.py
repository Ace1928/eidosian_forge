import logging
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import networkx_available, matplotlib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import TempfileManager
from pyomo.environ import (
from pyomo.contrib.community_detection.detection import (
from pyomo.contrib.community_detection.community_graph import generate_model_graph
from pyomo.solvers.tests.models.LP_unbounded import LP_unbounded
from pyomo.solvers.tests.models.QP_simple import QP_simple
from pyomo.solvers.tests.models.LP_inactive_index import LP_inactive_index
from pyomo.solvers.tests.models.SOS1_simple import SOS1_simple
@unittest.skipUnless(community_louvain_available, "'community' package from 'python-louvain' is not available.")
@unittest.skipUnless(networkx_available, 'networkx is not available.')
class TestDecomposition(unittest.TestCase):

    def test_communities_1(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2}, {3: 0, 4: 1, 5: 0, 6: 2, 7: 2, 8: 0, 9: 0, 10: 0, 11: 2, 12: 1, 13: 1, 14: 1}, {3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2}, {3: 0, 4: 1, 5: 0, 6: 0, 7: 0, 8: 2, 9: 2, 10: 2, 11: 0, 12: 1, 13: 1, 14: 1}, {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0}, {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2}, {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0, 12: 2, 13: 2, 14: 2}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0, 5: 1, 6: 1, 7: 1, 8: 0, 9: 2, 10: 2, 11: 2}, {0: 0, 1: 1, 2: 2, 3: 1, 4: 2, 5: 0, 6: 0, 7: 0, 8: 1, 9: 1, 10: 1, 11: 0, 12: 2, 13: 2, 14: 2}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 0, 8: 2}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 1, 5: 1, 6: 0, 7: 2}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1, 6: 1, 7: 0, 8: 2}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['obj[1]', 'OBJ', 'c1[3]', 'c1[4]', 'c2[1]'], ['x', 'y']), 1: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z']), 2: (['c1[1]', 'c1[2]', 'c2[2]'], ['x'])}", "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['obj[1]', 'OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x', 'y']), 1: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z']), 2: (['c1[3]', 'c1[4]', 'c2[1]'], ['y'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['obj[2]', 'c1[3]', 'c2[1]', 'B[2].c'], ['x', 'y', 'z']), 1: (['c1[2]', 'c2[2]'], ['x'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['obj[2]', 'c1[3]', 'c2[1]', 'B[2].c'], ['x', 'y', 'z']), 1: (['c1[2]', 'c2[2]'], ['x'])}", "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['obj[1]', 'obj[2]', 'OBJ', 'c1[1]', 'c1[2]', 'c1[3]', 'c1[4]', 'c2[1]', 'c2[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z'])}", "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['obj[1]', 'obj[2]', 'OBJ', 'c1[1]', 'c1[2]', 'c1[3]', 'c1[4]', 'c2[1]', 'c2[2]', 'b.c', 'B[1].c', 'B[2].c'], ['x', 'y', 'z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['obj[2]', 'c1[2]', 'c1[3]', 'c2[1]', 'c2[2]', 'B[2].c'], ['x', 'y', 'z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['obj[2]', 'c1[2]', 'c1[3]', 'c2[1]', 'c2[2]', 'B[2].c'], ['x', 'y', 'z'])}", "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['obj[1]', 'c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['OBJ', 'c1[1]', 'c1[2]', 'c2[2]'], ['x']), 1: (['obj[1]', 'c1[3]', 'c1[4]', 'c2[1]'], ['y']), 2: (['obj[2]', 'b.c', 'B[1].c', 'B[2].c'], ['z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['obj[2]', 'B[2].c'], ['z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['B[2].c'], ['z'])}", "{0: (['c1[2]', 'c2[2]'], ['x']), 1: (['c1[3]', 'c2[1]'], ['y']), 2: (['obj[2]', 'B[2].c'], ['z'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_2(self):
        m_class = QP_simple()
        m_class._generate_model()
        model = m = m_class.model
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{2: 0, 3: 0}, {2: 0, 3: 0, 4: 0, 5: 0}, {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0, 5: 0}, {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0}, {2: 0, 3: 0}, {2: 0, 3: 0, 4: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 0, 4: 1, 5: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}, {0: 0, 1: 1, 2: 1, 3: 0}, {0: 0, 1: 1, 2: 1, 3: 1, 4: 0}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['inactive_obj', 'obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c1', 'c2'], ['x', 'y'])}", "{0: (['obj', 'c1', 'c2'], ['x', 'y'])}", "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}", "{0: (['obj', 'c2'], ['x']), 1: (['inactive_obj', 'c1'], ['y'])}", "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}", "{0: (['obj', 'c2'], ['x']), 1: (['inactive_obj', 'c1'], ['y'])}", "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}", "{0: (['c2'], ['x']), 1: (['obj', 'c1'], ['y'])}", "{0: (['c2'], ['x']), 1: (['c1'], ['y'])}", "{0: (['c2'], ['x']), 1: (['obj', 'c1'], ['y'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_3(self):
        m_class = LP_unbounded()
        m_class._generate_model()
        model = m = m_class.model
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{}, {2: 0}, {}, {2: 0}, {}, {2: 0}, {}, {2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1}, {0: 0, 1: 0, 2: 0}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ['{}', "{0: (['o'], ['x', 'y'])}", '{}', "{0: (['o'], ['x', 'y'])}", '{}', "{0: (['o'], ['x', 'y'])}", '{}', "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}", "{0: ([], ['x']), 1: ([], ['y'])}", "{0: (['o'], ['x', 'y'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_4(self):
        m_class = SOS1_simple()
        m_class._generate_model()
        model = m = m_class.model
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{3: 0, 4: 1, 5: 0}, {3: 0, 4: 1, 5: 0, 6: 1}, {3: 0, 4: 1, 5: 0}, {3: 0, 4: 0, 5: 0, 6: 0}, {3: 0, 4: 1, 5: 0}, {3: 0, 4: 1, 5: 0, 6: 1}, {3: 0, 4: 1, 5: 0}, {3: 0, 4: 0, 5: 0, 6: 0}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 1}, {0: 0, 1: 0, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1}, {0: 0, 1: 1, 2: 2, 3: 2, 4: 0, 5: 1}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 2, 5: 0, 6: 1}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}", "{0: (['obj', 'c2'], ['x', 'y[1]', 'y[2]']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}", "{0: (['obj', 'c2'], ['x', 'y[1]', 'y[2]']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['c1', 'c4'], ['y[1]', 'y[2]']), 1: (['c2'], ['x'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c2'], ['x']), 1: (['c1', 'c4'], ['y[1]', 'y[2]'])}", "{0: (['obj', 'c1', 'c2', 'c4'], ['x', 'y[1]', 'y[2]'])}", "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}", "{0: (['obj', 'c2'], ['x']), 1: (['c4'], ['y[1]']), 2: (['c1'], ['y[2]'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_5(self):
        model = m = create_model_5()
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5}, {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}, {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 1, 7: 2, 8: 4, 9: 0, 10: 3, 11: 5}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['obj', 'c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), 3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), 3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), 3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}", "{0: (['c1', 'c2', 'c3', 'c4', 'c5'], ['i1', 'i2', 'i3', 'i4', 'i5', 'i6'])}", "{0: (['c3'], ['i1']), 1: (['obj'], ['i2']), 2: (['c1'], ['i3']), 3: (['c4'], ['i4']), 4: (['c2'], ['i5']), 5: (['c5'], ['i6'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_6(self):
        model = m = create_model_6()
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {4: 0, 5: 1}, {4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}", "{0: (['obj', 'c1'], ['x1', 'x2']), 1: (['c2'], ['x3', 'x4'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_7(self):
        model = m = disconnected_model()
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {2: 0}, {2: 0, 3: 1, 4: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}, {0: 0, 1: 1, 2: 0}, {0: 0, 1: 1, 2: 2, 3: 0, 4: 0}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1'])}", "{0: (['OBJ'], []), 1: (['obj', 'c1'], ['x1'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}", "{0: (['c1'], ['x1']), 1: ([], ['x2'])}", "{0: (['obj', 'c1'], ['x1']), 1: ([], ['x2']), 2: (['OBJ'], [])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_decode_1(self):
        model = m = decode_model_1()
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}, {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 1, 8: 1}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x2', 'x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2', 'c3'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}", "{0: (['c1', 'c2'], ['x1', 'x2']), 1: (['c3', 'c4', 'c5'], ['x3', 'x4'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_decode_2(self):
        model = m = decode_model_2()
        test_community_maps, test_partitions = _collect_community_maps(model)
        correct_partitions = [{7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 0, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 0, 8: 0, 9: 1, 10: 1, 11: 1, 12: 1}]
        if correct_partitions == test_partitions:
            str_test_community_maps = [str(community_map) for community_map in test_community_maps]
            correct_community_maps = ["{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), 1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), 1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), 1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]', 'x[4]', 'x[5]']), 1: (['c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[3]', 'x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2', 'c3'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}", "{0: (['c1', 'c2'], ['x[1]', 'x[2]', 'x[3]']), 1: (['c3', 'c4', 'c5', 'c6'], ['x[4]', 'x[5]', 'x[6]', 'x[7]'])}"]
            self.assertEqual(correct_community_maps, str_test_community_maps)
        correct_num_communities, correct_num_nodes, test_num_communities, test_num_nodes = _collect_partition_dependent_tests(test_community_maps, test_partitions)
        self.assertEqual(correct_num_communities, test_num_communities)
        self.assertEqual(correct_num_nodes, test_num_nodes)

    def test_communities_8(self):
        output = StringIO()
        with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.ERROR):
            detect_communities(ConcreteModel())
        self.assertIn('in detect_communities: Empty community map was returned', output.getvalue())
        with LoggingIntercept(output, 'pyomo.contrib.community_detection', logging.WARNING):
            detect_communities(one_community_model())
        self.assertIn('Community detection found that with the given parameters, the model could not be decomposed - only one community was found', output.getvalue())
        model = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid model: 'model=%s' - model must be an instance of ConcreteModel" % model):
            detect_communities(model)
        model = create_model_6()
        type_of_community_map = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for type_of_community_map: 'type_of_community_map=%s' - Valid values: 'bipartite', 'constraint', 'variable'" % type_of_community_map):
            detect_communities(model, type_of_community_map=type_of_community_map)
        with_objective = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for with_objective: 'with_objective=%s' - with_objective must be a Boolean" % with_objective):
            detect_communities(model, with_objective=with_objective)
        weighted_graph = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for weighted_graph: 'weighted_graph=%s' - weighted_graph must be a Boolean" % weighted_graph):
            detect_communities(model, weighted_graph=weighted_graph)
        random_seed = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed):
            detect_communities(model, random_seed=random_seed)
        random_seed = -1
        with self.assertRaisesRegex(ValueError, "Invalid value for random_seed: 'random_seed=%s' - random_seed must be a non-negative integer" % random_seed):
            detect_communities(model, random_seed=random_seed)
        use_only_active_components = 'foo'
        with self.assertRaisesRegex(TypeError, "Invalid value for use_only_active_components: 'use_only_active_components=%s' - use_only_active_components must be True or None" % use_only_active_components):
            detect_communities(model, use_only_active_components=use_only_active_components)

    @unittest.skipUnless(matplotlib_available, 'matplotlib is not available.')
    def test_visualize_model_graph_1(self):
        model = decode_model_1()
        community_map_object = detect_communities(model)
        with TempfileManager:
            fig, pos = community_map_object.visualize_model_graph(filename=TempfileManager.create_tempfile('test_visualize_model_graph_1.png'))
        correct_pos_dict_length = 5
        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)

    @unittest.skipUnless(matplotlib_available, 'matplotlib is not available.')
    def test_visualize_model_graph_2(self):
        model = decode_model_2()
        community_map_object = detect_communities(model)
        with TempfileManager:
            fig, pos = community_map_object.visualize_model_graph(type_of_graph='bipartite', filename=TempfileManager.create_tempfile('test_visualize_model_graph_2.png'))
        correct_pos_dict_length = 13
        self.assertTrue(isinstance(pos, dict))
        self.assertEqual(len(pos), correct_pos_dict_length)

    def test_generate_structured_model_1(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model
        community_map_object = cmo = detect_communities(model, random_seed=5)
        correct_partition = {3: 0, 4: 1, 5: 0, 6: 0, 7: 1, 8: 0}
        correct_components = {"b[0].'B[2].c'", "b[0].'c2[1]'", "b[0].'c1[3]'", 'equality_constraint_list[1]', "b[1].'c2[2]'", 'b[1].x', 'b[0].x', 'b[0].y', 'b[0].z', "b[0].'obj[2]'", "b[1].'c1[2]'"}
        structured_model = cmo.generate_structured_model()
        self.assertIsInstance(structured_model, Block)
        all_components = set([str(component) for component in structured_model.component_data_objects(ctype=(Var, Constraint, Objective, ConstraintList), active=cmo.use_only_active_components, descend_into=True)])
        if cmo.graph_partition == correct_partition:
            self.assertEqual(2, len(cmo.community_map), len(list(structured_model.component_data_objects(ctype=Block, descend_into=True))))
            self.assertEqual(all_components, correct_components)
            for objective in structured_model.component_data_objects(ctype=Objective, descend_into=True):
                objective_expr = str(objective.expr)
            correct_objective_expr = '- b[0].x + b[0].y + b[0].z'
            self.assertEqual(correct_objective_expr, objective_expr)
        self.assertEqual(len(correct_partition), len(cmo.graph_partition))
        self.assertEqual(len(correct_components), len(all_components))

    def test_generate_structured_model_2(self):
        m_class = LP_inactive_index()
        m_class._generate_model()
        model = m = m_class.model
        community_map_object = cmo = detect_communities(model, with_objective=False, random_seed=5)
        correct_partition = {3: 0, 4: 1, 5: 1, 6: 0, 7: 2}
        correct_components = {'b[2].B[2].c', 'b[1].y', 'z', 'b[0].c1[2]', 'b[1].c1[3]', 'obj[2]', 'equality_constraint_list[3]', 'b[0].x', 'b[1].c2[1]', 'b[2].z', 'x', 'equality_constraint_list[1]', 'b[0].c2[2]', 'y', 'equality_constraint_list[2]'}
        structured_model = cmo.generate_structured_model()
        self.assertIsInstance(structured_model, Block)
        all_components = set([str(component) for component in structured_model.component_data_objects(ctype=(Var, Constraint, Objective, ConstraintList), active=cmo.use_only_active_components, descend_into=True)])
        if cmo.graph_partition == correct_partition:
            self.assertEqual(3, len(cmo.community_map), len(list(structured_model.component_data_objects(ctype=Block, descend_into=True))))
            self.assertEqual(correct_components, all_components)
            for objective in structured_model.component_data_objects(ctype=Objective, descend_into=True):
                objective_expr = str(objective.expr)
            correct_objective_expr = '- x + y + z'
            self.assertEqual(correct_objective_expr, objective_expr)
        self.assertEqual(len(correct_partition), len(cmo.graph_partition))
        self.assertEqual(len(correct_components), len(all_components))